"""Project-wide pytest policy and opt-in PostgreSQL configuration."""

import os

import pytest
from sqlalchemy.engine import make_url


def pytest_addoption(parser):
    parser.addini(
        "test_name_max_underscores",
        "maximum underscore separators allowed in a test function name",
        default="3",
    )
    parser.addini(
        "postgresql_url_env",
        "environment variable containing the PostgreSQL integration-test URL",
        default="LLMPERF_TEST_DB",
    )


def pytest_collection_modifyitems(config, items):
    """Enforce concise names and skip opt-in database tests when unconfigured."""

    maximum = int(config.getini("test_name_max_underscores"))
    database_environment = config.getini("postgresql_url_env")
    database_configured = bool(os.environ.get(database_environment))
    invalid_names = []
    missing_fixtures = []

    for item in items:
        name = getattr(item, "originalname", None) or item.name.split("[", 1)[0]
        if name.startswith("test_") and name.count("_") > maximum:
            invalid_names.append(f"{item.nodeid}: {name}")

        if item.get_closest_marker("postgresql") is None:
            continue
        if "postgresql_url" not in item.fixturenames:
            missing_fixtures.append(item.nodeid)
        if not database_configured:
            item.add_marker(
                pytest.mark.skip(
                    reason=f"{database_environment} is not explicitly configured"
                )
            )

    errors = []
    if invalid_names:
        errors.append(
            f"test names may contain at most {maximum} underscores:\n  "
            + "\n  ".join(invalid_names)
        )
    if missing_fixtures:
        errors.append(
            "postgresql tests must request the postgresql_url fixture:\n  "
            + "\n  ".join(missing_fixtures)
        )
    if errors:
        raise pytest.UsageError("\n".join(errors))


@pytest.fixture(scope="session")
def postgresql_url(request):
    """Return a validated, explicitly configured disposable PostgreSQL URL."""

    environment = request.config.getini("postgresql_url_env")
    value = os.environ.get(environment)
    if not value:
        pytest.skip(f"{environment} is not explicitly configured")
    try:
        url = make_url(value)
    except Exception as exc:
        pytest.fail(f"{environment} is not a valid SQLAlchemy URL: {exc}")
    if url.drivername != "postgresql+asyncpg":
        pytest.fail(f"{environment} must use the postgresql+asyncpg driver")
    database_name = url.database or ""
    if "test" not in database_name.lower():
        pytest.fail(
            f"{environment} must select a disposable database whose name contains 'test'"
        )
    return value
