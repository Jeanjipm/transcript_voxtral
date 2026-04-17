"""Détection du champ de saisie éditable via l'API Accessibility macOS.

Utilise `AXUIElementCreateSystemWide` + `kAXFocusedUIElementAttribute` pour
récupérer l'élément UI actuellement au focus au niveau système, puis regarde
son `kAXRoleAttribute` pour décider si c'est un champ de texte.

Permission requise : Accessibility (la même que pour pynput) — pas de prompt
supplémentaire pour l'utilisateur.
"""

from __future__ import annotations

from ApplicationServices import (
    AXUIElementCopyAttributeValue,
    AXUIElementCreateSystemWide,
    kAXFocusedUIElementAttribute,
    kAXRoleAttribute,
)


_EDITABLE_ROLES: set[str] = {
    "AXTextField",
    "AXTextArea",
    "AXComboBox",
    "AXSearchField",
    "AXSecureTextField",
}


def is_editable_field_focused() -> bool:
    """True si l'élément UI focus est un champ de saisie éditable.

    Fallback permissif : si l'introspection AX échoue (app non-accessible
    type Electron/Java, ou transient system state), on retourne True pour
    laisser le paste se faire — mieux vaut coller quelque part qui encaisse
    pas que bloquer à tort quand on est en fait dans un éditeur valide.
    """
    system = AXUIElementCreateSystemWide()
    err, focused = AXUIElementCopyAttributeValue(
        system, kAXFocusedUIElementAttribute, None
    )
    if err != 0 or focused is None:
        return True
    err, role = AXUIElementCopyAttributeValue(focused, kAXRoleAttribute, None)
    if err != 0 or role is None:
        return True
    return str(role) in _EDITABLE_ROLES
