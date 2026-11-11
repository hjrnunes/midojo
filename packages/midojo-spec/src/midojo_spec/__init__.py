"""The MiDojo contracts, on the standard library alone.

``midojo_spec`` holds what MiDojo, the interceptor service, a consumer and Garak must read
the same way, so each reads one implementation. It imports nothing outside the standard
library and runs on Python 3.11 and later. Import the module you need; this package
imports none of them.

- ``judge``: the ``midojo.judge.v1`` request and response rules.
- ``judge_exchange``: the record a judge service keeps of each model request.
- ``rubric_judge``: the rubric judge, a port of Garak's, and its HTTP call.
"""
