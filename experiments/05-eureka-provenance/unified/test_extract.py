"""Extraction-rule regression tests — no model, no GPU.  python unified\\test_extract.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract import explicit_fact, may_state_fact

# explicit whole-message templates -> user-taught
assert explicit_fact("my name is Eric")[:2] == ("What is my name", "Eric")
assert explicit_fact("My favourite drink is apple juice.")[:2] == ("What is my favorite drink", "apple juice")
assert explicit_fact("remember that the meeting is on Friday")[:2] == ("What is the meeting", "on Friday")

# the measured failures: no longer eaten by templates
assert explicit_fact("my name is not important") is None
assert explicit_fact("which colour do I like best?") is None
assert explicit_fact("I like the way you answer") is None
assert explicit_fact("remember when the server is down") is None

# what the core's extractor may be asked about
assert may_state_fact("my sister is called Anna", "")
assert may_state_fact("my favoirte food is apple", "")
assert may_state_fact("basketball", "You haven't mentioned a favorite sport yet.")
assert not may_state_fact("basketball", "")
assert not may_state_fact("tell me something short about basketball", "ok")
assert not may_state_fact("write a python function that reverses a string", "ok")
assert not may_state_fact("translate 'good morning, how are you' to French", "ok")
assert not may_state_fact("what's my favorite sport?", "ok")
print("extract tests: ok")
