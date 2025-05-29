import pytest
from pydantic import ValidationError
from bayes_mcp.schemas.mcp_schemas import (
    Variable, 
    CreateModelRequest, 
    UpdateBeliefRequest, 
    PredictRequest
)


class TestVariable:
    """Test suite for Variable schema."""
    
    def test_valid_beta_distribution(self):
        """Test valid beta distribution variable."""
        variable = Variable(
            name="p",
            distribution="beta",
            params={"alpha": 1, "beta": 1}
        )
        assert variable.name == "p"
        assert variable.distribution == "beta"
        assert variable.params["alpha"] == 1
        assert variable.params["beta"] == 1

    def test_valid_normal_distribution(self):
        """Test valid normal distribution variable."""
        variable = Variable(
            name="mu",
            distribution="normal",
            params={"mu": 0, "sigma": 1}
        )
        assert variable.name == "mu"
        assert variable.distribution == "normal"
        assert variable.params["mu"] == 0
        assert variable.params["sigma"] == 1

    def test_valid_gamma_distribution(self):
        """Test valid gamma distribution variable."""
        variable = Variable(
            name="gamma_var",
            distribution="gamma",
            params={"alpha": 2, "beta": 1}
        )
        assert variable.name == "gamma_var"
        assert variable.distribution == "gamma"
        assert variable.params["alpha"] == 2

    def test_valid_binomial_distribution(self):
        """Test valid binomial distribution variable."""
        variable = Variable(
            name="observations",
            distribution="binomial",
            params={"n": 10, "p": 0.5}
        )
        assert variable.name == "observations"
        assert variable.distribution == "binomial"
        assert variable.params["n"] == 10
        assert variable.params["p"] == 0.5

    def test_binomial_with_variable_reference(self):
        """Test binomial distribution with variable reference."""
        variable = Variable(
            name="trials",
            distribution="binomial",
            params={"n": 10, "p": "success_rate"}
        )
        assert variable.params["p"] == "success_rate"

    def test_observed_data_integer(self):
        """Test variable with observed integer data."""
        variable = Variable(
            name="data",
            distribution="binomial",
            params={"n": 10, "p": 0.5},
            observed=7
        )
        assert variable.observed == 7

    def test_observed_data_list(self):
        """Test variable with observed list data."""
        data = [1, 2, 3, 4, 5]
        variable = Variable(
            name="samples",
            distribution="normal",
            params={"mu": 0, "sigma": 1},
            observed=data
        )
        assert variable.observed == data

    def test_missing_name_invalid(self):
        """Test that missing name is invalid."""
        with pytest.raises(ValidationError):
            Variable(
                distribution="normal",
                params={"mu": 0, "sigma": 1}
            )

    def test_empty_name_invalid(self):
        """Test that empty name is invalid."""
        with pytest.raises(ValidationError):
            Variable(
                name="",
                distribution="normal",
                params={"mu": 0, "sigma": 1}
            )

    def test_extra_params_allowed(self):
        """Test that extra parameters don't cause validation errors."""
        variable = Variable(
            name="test_var",
            distribution="normal",
            params={"mu": 0, "sigma": 1, "extra_param": "ignored"}
        )
        assert "extra_param" in variable.params


class TestCreateModelRequest:
    """Test suite for CreateModelRequest schema."""

    def test_valid_create_model_request(self):
        """Test valid create model request."""
        request = CreateModelRequest(
            model_name="test_model",
            variables={
                "p": {
                    "distribution": "beta",
                    "params": {"alpha": 1, "beta": 1}
                },
                "observations": {
                    "distribution": "binomial",
                    "params": {"n": 10, "p": "p"},
                    "observed": 7
                }
            }
        )
        assert request.model_name == "test_model"
        assert len(request.variables) == 2
        assert "p" in request.variables
        assert "observations" in request.variables

    def test_empty_model_name_invalid(self):
        """Test that empty model name is invalid."""
        with pytest.raises(ValidationError):
            CreateModelRequest(
                model_name="",
                variables={"p": {"distribution": "beta", "params": {"alpha": 1, "beta": 1}}}
            )

    def test_empty_variables_invalid(self):
        """Test that empty variables dictionary is invalid."""
        with pytest.raises(ValidationError):
            CreateModelRequest(
                model_name="test_model",
                variables={}
            )

    def test_variable_name_validation(self):
        """Test variable name validation."""
        # Valid variable names
        valid_names = ["var1", "variable_name", "CamelCase", "snake_case"]
        for name in valid_names:
            request = CreateModelRequest(
                model_name="test",
                variables={name: {"distribution": "beta", "params": {"alpha": 1, "beta": 1}}}
            )
            assert name in request.variables

    def test_model_name_special_characters(self):
        """Test model names with special characters."""
        special_names = ["model-1", "model.test", "model_name"]
        for name in special_names:
            request = CreateModelRequest(
                model_name=name,
                variables={"p": {"distribution": "beta", "params": {"alpha": 1, "beta": 1}}}
            )
            assert request.model_name == name


class TestUpdateBeliefRequest:
    """Test suite for UpdateBeliefRequest schema."""

    def test_valid_update_request(self):
        """Test valid update belief request."""
        request = UpdateBeliefRequest(
            model_name="test_model",
            evidence={"observed_value": 7},
            sample_kwargs={"draws": 500, "chains": 2}
        )
        assert request.model_name == "test_model"
        assert request.evidence["observed_value"] == 7
        assert request.sample_kwargs["draws"] == 500

    def test_empty_evidence_allowed(self):
        """Test that empty evidence dictionary is allowed."""
        request = UpdateBeliefRequest(
            model_name="test_model",
            evidence={}
        )
        assert request.evidence == {}

    def test_optional_sample_kwargs(self):
        """Test that sample_kwargs can be optional."""
        request = UpdateBeliefRequest(
            model_name="test_model",
            evidence={"data": 5}
        )
        assert request.model_name == "test_model"
        assert request.sample_kwargs is None

    def test_empty_model_name_invalid(self):
        """Test that empty model name is invalid."""
        with pytest.raises(ValidationError):
            UpdateBeliefRequest(
                model_name="",
                evidence={"data": 5}
            )


class TestPredictRequest:
    """Test suite for PredictRequest schema."""

    def test_valid_predict_request(self):
        """Test valid predict request."""
        request = PredictRequest(
            model_name="test_model",
            variables=["p", "mu"],
            conditions={"evidence": "data"}
        )
        assert request.model_name == "test_model"
        assert "p" in request.variables
        assert "mu" in request.variables

    def test_empty_variables_invalid(self):
        """Test that empty variables list is invalid."""
        with pytest.raises(ValidationError):
            PredictRequest(
                model_name="test",
                variables=[]
            )

    def test_optional_conditions(self):
        """Test that conditions can be optional."""
        request = PredictRequest(
            model_name="test_model",
            variables=["p"]
        )
        assert request.conditions is None


# Remove the response schema tests since they don't exist in current schemas


class TestSchemaIntegration:
    """Integration tests for schema validation."""

    def test_end_to_end_model_creation(self):
        """Test complete model creation schema validation."""
        # Create a complex model request
        request_data = {
            "model_name": "ab_test_model",
            "variables": {
                "conversion_rate_a": {
                    "distribution": "beta",
                    "params": {"alpha": 1, "beta": 1}
                },
                "conversion_rate_b": {
                    "distribution": "beta",
                    "params": {"alpha": 1, "beta": 1}
                },
                "conversions_a": {
                    "distribution": "binomial",
                    "params": {"n": 100, "p": "conversion_rate_a"},
                    "observed": 12
                },
                "conversions_b": {
                    "distribution": "binomial",
                    "params": {"n": 100, "p": "conversion_rate_b"},
                    "observed": 18
                }
            }
        }
        
        # Should validate successfully
        request = CreateModelRequest(**request_data)
        assert request.model_name == "ab_test_model"
        assert len(request.variables) == 4

    def test_end_to_end_belief_update(self):
        """Test complete belief update schema validation."""
        update_data = {
            "model_name": "test_model",
            "evidence": {
                "new_observations": [1, 1, 0, 1, 0],
                "additional_data": 15
            },
            "sample_kwargs": {
                "draws": 2000,
                "tune": 1000,
                "chains": 2,
                "progressbar": False
            }
        }
        
        # Should validate successfully
        request = UpdateBeliefRequest(**update_data)
        assert request.model_name == "test_model"
        assert request.sample_kwargs["draws"] == 2000

    def test_schema_serialization(self):
        """Test that schemas can be serialized to/from JSON."""
        request = CreateModelRequest(
            model_name="test",
            variables={"p": {"distribution": "beta", "params": {"alpha": 1, "beta": 1}}}
        )
        
        # Should be able to convert to dict
        request_dict = request.model_dump()
        assert request_dict["model_name"] == "test"
        
        # Should be able to recreate from dict
        new_request = CreateModelRequest(**request_dict)
        assert new_request.model_name == request.model_name