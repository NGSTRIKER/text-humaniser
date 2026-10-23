"""Allow ``python -m humanizer`` to invoke the CLI."""

from humanizer.main import main

raise SystemExit(main())
