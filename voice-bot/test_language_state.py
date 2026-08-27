import unittest

from language_state import LanguageState


class TestLanguageState(unittest.TestCase):
    def test_defaults_to_english(self):
        state = LanguageState()
        self.assertEqual(state.current_language, "en")
        self.assertFalse(state.established)

    def test_first_reliable_hindi_turn_switches_immediately(self):
        state = LanguageState()
        lang, switched = state.observe_stt(
            "hi-IN",
            0.94,
        )
        self.assertEqual(lang, "hi")
        self.assertTrue(switched)
        self.assertEqual(
            state.switch_reason,
            "initial_detection",
        )

    def test_first_reliable_telugu_turn_switches_immediately(self):
        state = LanguageState()
        lang, switched = state.observe_stt(
            "te-IN",
            0.96,
        )
        self.assertEqual(lang, "te")
        self.assertTrue(switched)

    def test_low_confidence_detection_does_not_switch(self):
        state = LanguageState()
        lang, switched = state.observe_stt(
            "te-IN",
            0.50,
        )
        self.assertEqual(lang, "en")
        self.assertFalse(switched)

    def test_sustained_change_requires_two_turns(self):
        state = LanguageState()
        state.observe_stt("hi-IN", 0.95)
        self.assertEqual(state.current_language, "hi")

        lang, switched = state.observe_stt("en-IN", 0.94)
        self.assertEqual(lang, "hi")
        self.assertFalse(switched)

        lang, switched = state.observe_stt("en-IN", 0.96)
        self.assertEqual(lang, "en")
        self.assertTrue(switched)
        self.assertEqual(
            state.switch_reason,
            "sustained_change",
        )

    def test_explicit_language_change_is_immediate(self):
        state = LanguageState()
        state.observe_stt("en-IN", 0.96)

        lang, switched = state.set_explicit(
            "te",
            "caller explicitly requested Telugu",
        )

        self.assertEqual(lang, "te")
        self.assertTrue(switched)
        self.assertEqual(
            state.switch_reason,
            "explicit_request",
        )

    def test_code_mixed_words_do_not_affect_state(self):
        # LanguageState does not inspect transcript text at all.
        # Code-mix is a speech-recognition/model responsibility.
        state = LanguageState()
        state.observe_stt("hi-IN", 0.95)

        self.assertEqual(
            state.current_language,
            "hi",
        )

    def test_call_isolation(self):
        a = LanguageState()
        b = LanguageState()

        a.observe_stt("te-IN", 0.95)

        self.assertEqual(a.current_language, "te")
        self.assertEqual(b.current_language, "en")
        self.assertEqual(b.turn_index, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)