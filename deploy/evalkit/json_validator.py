"""
Tiny inline JSON-Schema-like validator (no jsonschema dependency, per the brief).

Supports the subset we need for json_schema items: type, required,
properties, items, minItems/maxItems, minLength/maxLength, minimum/maximum,
pattern, enum. Returns a list of human-readable error strings; empty list
means the object validated.
"""
import re


def _check_type(obj, t):
    if t == "object":
        return isinstance(obj, dict)
    if t == "array":
        return isinstance(obj, list)
    if t == "string":
        return isinstance(obj, str)
    if t == "integer":
        return isinstance(obj, int) and not isinstance(obj, bool)
    if t == "number":
        return isinstance(obj, (int, float)) and not isinstance(obj, bool)
    if t == "boolean":
        return isinstance(obj, bool)
    if t == "null":
        return obj is None
    return True  # unknown type name: don't block on it


def validate_schema(obj, schema, path="$"):
    errors = []
    t = schema.get("type")

    if t is not None:
        if not _check_type(obj, t):
            errors.append(f"{path}: expected type '{t}', got {type(obj).__name__}")
            return errors  # type mismatch makes deeper checks meaningless

    if t == "object":
        if not isinstance(obj, dict):
            return errors
        for req in schema.get("required", []):
            if req not in obj:
                errors.append(f"{path}: missing required key '{req}'")
        props = schema.get("properties", {})
        for key, subschema in props.items():
            if key in obj:
                errors.extend(validate_schema(obj[key], subschema, f"{path}.{key}"))
        # HOLE FIX: there was no way to reject extra/unexpected keys, so a
        # response padding the object with bogus extra fields -- or putting
        # a wrong value under a key that isn't in `properties` at all --
        # always validated cleanly even though several prompts explicitly
        # say "exactly these fields". additionalProperties is opt-in
        # (default permissive, matching prior behavior) so existing schemas
        # that don't set it are unaffected.
        if schema.get("additionalProperties") is False:
            extra = sorted(k for k in obj if k not in props)
            if extra:
                errors.append(f"{path}: unexpected additional key(s) {extra!r} (additionalProperties: false)")

    elif t == "array":
        if "minItems" in schema and len(obj) < schema["minItems"]:
            errors.append(f"{path}: length {len(obj)} < minItems {schema['minItems']}")
        if "maxItems" in schema and len(obj) > schema["maxItems"]:
            errors.append(f"{path}: length {len(obj)} > maxItems {schema['maxItems']}")
        items_schema = schema.get("items")
        if items_schema:
            for i, el in enumerate(obj):
                errors.extend(validate_schema(el, items_schema, f"{path}[{i}]"))

    elif t == "string":
        if "minLength" in schema and len(obj) < schema["minLength"]:
            errors.append(f"{path}: length {len(obj)} < minLength {schema['minLength']}")
        if "maxLength" in schema and len(obj) > schema["maxLength"]:
            errors.append(f"{path}: length {len(obj)} > maxLength {schema['maxLength']}")
        if "pattern" in schema and re.search(schema["pattern"], obj) is None:
            errors.append(f"{path}: {obj!r} does not match pattern {schema['pattern']!r}")

    elif t in ("integer", "number"):
        if "minimum" in schema and obj < schema["minimum"]:
            errors.append(f"{path}: {obj} < minimum {schema['minimum']}")
        if "maximum" in schema and obj > schema["maximum"]:
            errors.append(f"{path}: {obj} > maximum {schema['maximum']}")

    if "enum" in schema and obj not in schema["enum"]:
        errors.append(f"{path}: {obj!r} not in enum {schema['enum']}")

    return errors
