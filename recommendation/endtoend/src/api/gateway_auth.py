"""Gateway가 검증해 전달한 회원 식별자만 추천 요청의 주체로 사용한다."""
import hmac
import os

from fastapi import HTTPException, Request


def authenticated_member_id(request: Request) -> int:
    expected = os.environ.get("INTERNAL_GATEWAY_SECRET")
    if not expected:
        raise HTTPException(status_code=503, detail="SERVICE_AUTH_NOT_CONFIGURED")
    secrets = request.headers.getlist("X-Internal-Secret")
    members = request.headers.getlist("X-Member-Id")
    if len(secrets) != 1 or not hmac.compare_digest(secrets[0].encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="SERVICE_UNAUTHORIZED")
    if (len(members) != 1 or not members[0].isascii() or not members[0].isdecimal()
            or len(members[0]) > 19 or not 0 < int(members[0]) <= 9223372036854775807):
        raise HTTPException(status_code=401, detail="SERVICE_UNAUTHORIZED")
    return int(members[0])
