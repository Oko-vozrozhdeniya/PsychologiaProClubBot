"""Fail-closed access boundary for private Telegram memory integration.

The bot must not infer consent from Telegram identity, club membership,
administrator status or an environment flag. A trusted external provider
must return explicit, independently verified permission objects.
"""

from datetime import datetime
from memory.coordinator import AccessPolicy, SourcePermission
from memory.private_integration import prepare_private_memory


def prepare_with_verified_access(
    *,
    user_id,
    message_id,
    text,
    timestamp,
    permission_provider,
    db_path=None,
):
    """Use long-term memory only after a trusted provider verifies access.

    The provider is a callable receiving (user_id, timestamp) and returning
    (AccessPolicy, sequence[SourcePermission]). No provider means no access.
    """
    if type(user_id) is not int or user_id <= 0:
        raise ValueError("Invalid Telegram user")
    if not isinstance(timestamp, datetime) or timestamp.tzinfo is None:
        raise ValueError("Timezone-aware timestamp required")
    if not callable(permission_provider):
        raise PermissionError("Trusted permission provider required")

    policy, permissions = permission_provider(user_id, timestamp)

    if not isinstance(policy, AccessPolicy) or policy.user_id != user_id:
        raise PermissionError("Verified matching policy required")
    if not (policy.verified and policy.consent and policy.entitled):
        raise PermissionError("Explicit consent and entitlement required")
    if not isinstance(permissions, (tuple, list)) or not permissions:
        raise PermissionError("Verified source permissions required")
    if not all(
        isinstance(item, SourcePermission) and item.verified
        for item in permissions
    ):
        raise PermissionError("Unverified source permissions")

    return prepare_private_memory(
        user_id=user_id,
        message_id=message_id,
        text=text,
        timestamp=timestamp,
        policy=policy,
        source_permissions=permissions,
        db_path=db_path,
    )
