"""Compatibility wrapper around the shared final-answer parser."""
from mobilevla.actions import ACTIONS, parse_answer


class ActionExtractor:
    action_space = list(ACTIONS)

    def extract(self, text):
        parsed = parse_answer(text)
        return {'velocity': list(parsed.velocity) if parsed.velocity is not None else None,
                'action': parsed.action}

    def extract_velocity_vector(self, text):
        return self.extract(text)['velocity']

    def extract_action(self, text):
        return self.extract(text)['action']
