#!/usr/bin/env python3
"""Plan or execute the single permitted fallback GPU micro-pilot."""

import sys

from story_projection_onto.fallback_acceptance import (
    establish_fallback_orchestrator_process_group,
    main,
)

if __name__ == "__main__":
    establish_fallback_orchestrator_process_group(sys.argv[1:])
    raise SystemExit(main(sys.argv[1:]))
