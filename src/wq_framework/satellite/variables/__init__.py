"""Concrete satellite variable retrievers.

Importing this package registers every shipped retriever with the registry
as a side effect, so `get_satellite_variable("precipitation")` works after
only `import wq_framework.satellite`.

Neither import below requires `earthengine-api` or the Google API client to
be installed — those are imported lazily and guarded, so the offline
("manual") mode works on a bare install.
"""

from . import manual as _manual  # noqa: F401 - registers "manual"
from . import gee_era5land as _gee_era5land  # noqa: F401 - registers the 5 ERA5-Land variables
