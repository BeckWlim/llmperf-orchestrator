"""Backend HTTP client with no dependency on backend implementation modules."""

from datetime import date, time
import json
import logging
from pathlib import Path
import socket
from time import monotonic
from typing import Any, Callable, Dict, List, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


LOGGER = logging.getLogger("llmperfctl.http")


class ClientError(RuntimeError):
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


def _json_default(value: Any) -> Any:
    """Serialize YAML-native temporal values at the HTTP JSON boundary."""

    if isinstance(value, (date, time)):
        return value.isoformat()
    raise TypeError(
        f"Object of type {value.__class__.__name__} is not JSON serializable"
    )


class LLMPerfClient:
    def __init__(
        self,
        base_url: str,
        token: Optional[str] = None,
        timeout: float = 30.0,
        token_provider: Optional[Callable[[], str]] = None,
        token_providers: Optional[Sequence[Callable[[], str]]] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.token_provider = token_provider
        self.token_providers = list(token_providers or [])
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        query: Optional[Dict[str, Any]] = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        if query:
            filtered = {key: value for key, value in query.items() if value is not None}
            url = f"{url}?{urlencode(filtered)}"
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, default=_json_default).encode("utf-8")
            headers["Content-Type"] = "application/json"
        providers: List[Optional[Callable[[], str]]]
        if self.token_provider is not None:
            providers = [self.token_provider]
        elif self.token is not None:
            providers = [lambda: self.token or ""]
        elif self.token_providers:
            providers = list(self.token_providers)
        else:
            providers = [None]

        for index, provider in enumerate(providers):
            attempt_headers = dict(headers)
            token = provider() if provider is not None else None
            if token:
                attempt_headers["Authorization"] = f"Bearer {token}"
            request = Request(
                url,
                data=body,
                headers=attempt_headers,
                method=method,
            )
            started = monotonic()
            LOGGER.debug(
                "HTTP request started: method=%s path=%s auth_candidate=%d/%d",
                method,
                path,
                index + 1,
                len(providers),
            )
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    content = response.read()
                    LOGGER.debug(
                        "HTTP request completed: method=%s path=%s status=%s "
                        "elapsed=%.3fs bytes=%d",
                        method,
                        path,
                        getattr(response, "status", 200),
                        monotonic() - started,
                        len(content),
                    )
                    if (
                        self.token_provider is None
                        and self.token is None
                        and provider is not None
                        and index > 0
                    ):
                        self.token_providers.remove(provider)
                        self.token_providers.insert(0, provider)
                    return json.loads(content) if content else None
            except HTTPError as exc:
                content = exc.read().decode("utf-8", errors="replace")
                try:
                    detail = json.loads(content).get("detail", content)
                except json.JSONDecodeError:
                    detail = content
                if exc.code == 401 and index + 1 < len(providers):
                    LOGGER.debug(
                        "HTTP authentication candidate rejected: method=%s path=%s "
                        "elapsed=%.3fs",
                        method,
                        path,
                        monotonic() - started,
                    )
                    continue
                raise ClientError(
                    f"HTTP {exc.code}: {detail}", status_code=exc.code
                ) from exc
            except (TimeoutError, socket.timeout) as exc:
                raise ClientError(
                    f"Backend request timed out after {self.timeout:g} seconds "
                    f"while waiting for {method} {path}. The backend may still be "
                    "processing it; inspect current state before retrying. Increase "
                    "the limit with 'llmperfctl --request-timeout SECONDS ...'."
                ) from exc
            except URLError as exc:
                raise ClientError(
                    f"Unable to reach {self.base_url}: {exc.reason}"
                ) from exc

        raise ClientError("No authentication candidate completed the request")

    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/health")

    def get_scheduler_status(self) -> Dict[str, Any]:
        return self._request("GET", "/api/v1/scheduler/status")

    def get_planner_status(self) -> Dict[str, Any]:
        return self._request("GET", "/api/v1/planner/status")

    def list_providers(self) -> Dict[str, Any]:
        return self._request("GET", "/api/v1/providers")

    def list_provider_models(
        self, provider_id: str, refresh: bool = False
    ) -> Dict[str, Any]:
        return self._request(
            "GET",
            f"/api/v1/providers/{quote(provider_id, safe='')}/models",
            query={"refresh": str(refresh).lower()},
        )

    def create_campaign(
        self, name: str, description: Optional[str], tags: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            "/api/v1/campaigns",
            {"name": name, "description": description, "tags": tags},
        )

    def list_campaigns(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        return self._request(
            "GET", "/api/v1/campaigns", query={"limit": limit, "offset": offset}
        )

    def get_campaign(self, campaign_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/api/v1/campaigns/{campaign_id}")

    def cancel_campaign(self, campaign_id: str) -> Dict[str, Any]:
        return self._request("POST", f"/api/v1/campaigns/{campaign_id}/cancel")

    def start_runner(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/api/v1/runners", payload)

    def start_campaign_runners(
        self, campaign_id: str, runners: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            f"/api/v1/campaigns/{campaign_id}/runners",
            {"runners": runners},
        )

    def start_campaign(
        self,
        campaign: Dict[str, Any],
        runners: List[Dict[str, Any]],
        runner_plans: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            "/api/v1/campaigns/start",
            {
                "campaign": campaign,
                "runners": runners,
                "runner_plans": runner_plans,
            },
        )

    def preview_runner_plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/api/v1/runner-plans/preview", payload)

    def create_runner_plan(
        self, campaign_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self._request(
            "POST", f"/api/v1/campaigns/{campaign_id}/runner-plans", payload
        )

    def list_runner_plans(
        self,
        status: Optional[str] = None,
        campaign_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        return self._request(
            "GET",
            "/api/v1/runner-plans",
            query={
                "status": status,
                "campaign_id": campaign_id,
                "limit": limit,
                "offset": offset,
            },
        )

    def get_runner_plan(self, runner_plan_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/api/v1/runner-plans/{runner_plan_id}")

    def get_runner_plan_events(self, runner_plan_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/api/v1/runner-plans/{runner_plan_id}/events")

    def change_runner_plan(self, runner_plan_id: str, action: str) -> Dict[str, Any]:
        return self._request("POST", f"/api/v1/runner-plans/{runner_plan_id}/{action}")

    def list_runners(
        self,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
        full: bool = False,
        campaign_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._request(
            "GET",
            "/api/v1/runners",
            query={
                "status": status,
                "limit": limit,
                "offset": offset,
                "full": str(full).lower(),
                "campaign_id": campaign_id,
            },
        )

    def get_runner(self, runner_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/api/v1/runners/{runner_id}")

    def cancel_runner(self, runner_id: str) -> Dict[str, Any]:
        return self._request("POST", f"/api/v1/runners/{runner_id}/cancel")

    def export_runner(self, runner_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/api/v1/runners/{runner_id}/export")

    def export_campaign(
        self, campaign_id: str, include_requests: bool = False
    ) -> Dict[str, Any]:
        return self._request(
            "GET",
            f"/api/v1/campaigns/{campaign_id}/export",
            query={"include_requests": str(include_requests).lower()},
        )

    def list_trusted_clients(self) -> Dict[str, Any]:
        return self._request("GET", "/api/v1/admin/trusted-clients")

    def write_trusted_client(
        self,
        username: str,
        public_key: str,
        role: str,
        display_name: Optional[str] = None,
        email: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._request(
            "PUT",
            f"/api/v1/admin/trusted-clients/{username}",
            {
                "public_key": public_key,
                "role": role,
                "display_name": display_name,
                "email": email,
            },
        )

    def revoke_trusted_client(self, username: str) -> Dict[str, Any]:
        return self._request("DELETE", f"/api/v1/admin/trusted-clients/{username}")

    def list_trusted_client_events(self, limit: int = 100) -> Dict[str, Any]:
        return self._request(
            "GET",
            "/api/v1/admin/trusted-client-events",
            query={"limit": limit},
        )


def write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
