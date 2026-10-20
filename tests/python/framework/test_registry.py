"""Contracts for public operator registration and implementation lookup."""

from unittest.mock import Mock

import pytest


def test_public_registration_preserves_function(registry):
    operator = Mock()

    assert registry.register_operator("test.op")(operator) is operator
    assert registry.get_op_name(operator) == "test.op"
    operator.assert_not_called()


def test_operator_name_is_explicit_not_inferred(registry):
    def misleading_name():
        pass

    registry.register_operator("norm.custom_name")(misleading_name)
    assert registry.get_op_name(misleading_name) == "norm.custom_name"


def test_duplicate_public_function_is_rejected_without_changing_name(registry):
    operator = Mock()
    registry.register_operator("test.first")(operator)

    with pytest.raises(ValueError, match="already registered"):
        registry.register_operator("test.second")(operator)

    assert registry.get_op_name(operator) == "test.first"


def test_duplicate_public_name_is_rejected_without_registering_function(registry):
    original, duplicate = Mock(), Mock()
    registry.register_operator("test.op")(original)

    with pytest.raises(ValueError, match="Duplicate public operator name"):
        registry.register_operator("test.op")(duplicate)

    assert registry.get_op_name(original) == "test.op"
    with pytest.raises(ValueError, match="Unregistered public operator"):
        registry.get_op_name(duplicate)


@pytest.mark.parametrize("query", ["get_op_name", "get_impl", "list_impls"])
def test_unregistered_function_is_rejected(registry, query):
    operator = Mock()
    arguments = (operator, "triton") if query == "get_impl" else (operator,)

    with pytest.raises(ValueError, match="Unregistered public operator"):
        getattr(registry, query)(*arguments)

    operator.assert_not_called()


def test_impl_registration_and_lookup_do_not_execute_backend(registry):
    operator, backend = Mock(), Mock()
    registry.register_operator("test.op")(operator)

    assert registry.register_impl("test.op", "triton")(backend) is backend
    assert registry.get_impl_by_name("test.op", "triton") is backend
    assert registry.get_impl(operator, "triton") is backend
    backend.assert_not_called()


def test_impl_can_be_registered_before_public_operator(registry):
    operator, backend = Mock(), Mock()
    registry.register_impl("test.op", "triton")(backend)
    registry.register_operator("test.op")(operator)

    assert registry.get_impl(operator, "triton") is backend


@pytest.mark.parametrize("reuse_callable", [False, True])
def test_duplicate_impl_is_rejected_without_overwriting(registry, reuse_callable):
    original, replacement = Mock(), Mock()
    registry.register_impl("test.op", "triton")(original)

    with pytest.raises(ValueError, match="Duplicate impl"):
        registry.register_impl("test.op", "triton")(
            original if reuse_callable else replacement
        )

    assert registry.get_impl_by_name("test.op", "triton") is original
    replacement.assert_not_called()


def test_same_impl_name_is_scoped_to_operator(registry):
    first, second = Mock(), Mock()
    registry.register_impl("test.first", "triton")(first)
    registry.register_impl("test.second", "triton")(second)

    assert registry.get_impl_by_name("test.first", "triton") is first
    assert registry.get_impl_by_name("test.second", "triton") is second


def test_aliases_preserve_callable_identity(registry):
    operator, backend = Mock(), Mock()
    registry.register_operator("test.op")(operator)
    registry.register_impl("test.op", "triton")(backend)
    registry.register_impl("test.op", "auto")(backend)

    assert registry.get_impl(operator, "auto") is registry.get_impl(operator, "triton")
    assert registry.list_impls(operator) == ["auto", "triton"]


def test_list_impls_is_sorted_and_excludes_other_operators(registry):
    operator = Mock()
    registry.register_operator("test.op")(operator)
    for impl in ("triton", "auto", "torch_native"):
        registry.register_impl("test.op", impl)(Mock())
    registry.register_impl("test.other", "ascendc")(Mock())

    expected = ["auto", "torch_native", "triton"]
    assert registry.list_impls_by_name("test.op") == expected
    assert registry.list_impls(operator) == expected
    # Mutating a returned list must not change subsequent enumeration.
    choices = registry.list_impls(operator)
    choices.clear()
    assert registry.list_impls(operator) == expected


def test_registered_operator_without_impls_has_no_choices(registry):
    operator = Mock()
    registry.register_operator("test.op")(operator)

    assert registry.list_impls(operator) == []
    with pytest.raises(registry.ImplementationNotFoundError, match=r"declared: \[\]"):
        registry.get_impl(operator, "auto")


def test_unknown_name_has_no_choices(registry):
    assert registry.list_impls_by_name("test.missing") == []
    with pytest.raises(registry.ImplementationNotFoundError, match=r"declared: \[\]"):
        registry.get_impl_by_name("test.missing", "triton")


@pytest.mark.parametrize("impl", ["missing", "TRITON", "", None])
def test_unknown_impl_reports_choices_without_fallback(registry, impl):
    operator, backend = Mock(), Mock()
    registry.register_operator("test.op")(operator)
    registry.register_impl("test.op", "auto")(backend)
    registry.register_impl("test.op", "triton")(backend)

    with pytest.raises(registry.ImplementationNotFoundError) as caught:
        registry.get_impl(operator, impl)

    message = str(caught.value)
    assert repr(impl) in message
    assert "test.op" in message
    assert "['auto', 'triton']" in message
    backend.assert_not_called()


def test_public_exports_use_the_same_registry(operator_modules):
    api, registry = operator_modules.public_api, operator_modules.registry
    operator, backend = Mock(), Mock()
    registry.register_operator("test.op")(operator)
    registry.register_impl("test.op", "custom")(backend)

    assert api.get_impl(operator, "custom") is backend
    assert api.list_impls(operator) == ["custom"]
