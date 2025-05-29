# Bayes MCP Project Refactoring Plan

## Project Overview

This plan outlines the complete refactoring of the Bayesian MCP server project to:
1. Rename `bayesian_mcp` module to `bayes_mcp` throughout the codebase
2. Implement comprehensive unit and integration tests
3. Ensure all tests pass successfully
4. Commit and push changes to GitHub

## Current State Analysis

### Project Structure
```
bayes/
├── bayesian_mcp/           # Main module (to be renamed)
│   ├── bayes_mcp/
│   ├── bayesian_engine/
│   ├── mcp/
│   ├── schemas/
│   └── utils/
├── demos/                  # Integration test sources
├── tests/                  # Empty (to be populated)
├── pyproject.toml         # Already correctly named "bayesian-mcp"
└── bayesian_mcp.py        # Entry point script
```

### Dependencies Found
- 17 files contain `bayesian_mcp` imports that need updating
- Demos provide excellent integration test templates
- Server startup/shutdown patterns already established

## Phase 1: Module Renaming Strategy

### 1.1 Directory Renaming
- Rename `bayesian_mcp/` → `bayes_mcp/`
- Rename `bayesian_mcp.py` → `bayes_mcp.py`

### 1.2 Import Updates Required
**Files requiring import changes:**
- `bayes_mcp/mcp/server.py`
- `debug_server.py`
- `demos/ab_testing_demo.py`
- `demos/financial_risk_demo.py` 
- `demos/master_demo.py`
- `demos/medical_diagnosis_demo.py`
- `demos/run_all_demos.py`
- `test_demo.py`

### 1.3 Configuration Updates
- Update `pyproject.toml` packages.find include pattern
- Update any remaining references in documentation

## Phase 2: Comprehensive Testing Strategy

### 2.1 Demo Analysis for Test Development
The `/Users/dionedge/devqai/bayes/demos` directory contains valuable integration test patterns:

**Existing Demos to Convert:**
- `ab_testing_demo.py` - A/B testing workflow with conversion rates
- `financial_risk_demo.py` - Portfolio risk assessment scenarios  
- `medical_diagnosis_demo.py` - Diagnostic probability calculations
- `master_demo.py` - Comprehensive demo orchestrator
- `run_all_demos.py` - Automated demo execution

**Demo Testing Patterns:**
- Server health checks before execution
- Multi-step Bayesian inference workflows
- Real-world scenario modeling
- Results validation and interpretation
- Error handling for server connectivity

### 2.2 Test Structure
```
tests/
├── conftest.py                    # Shared fixtures and configuration
├── unit/
│   ├── __init__.py
│   ├── test_bayesian_engine.py    # Core engine functionality
│   ├── test_mcp_handlers.py       # MCP request handlers
│   ├── test_schemas.py            # Pydantic schema validation
│   └── test_utils.py              # Utility functions
├── integration/
│   ├── __init__.py
│   ├── test_server_lifecycle.py   # Server start/stop/health
│   ├── test_ab_testing_flow.py    # Based on ab_testing_demo
│   ├── test_medical_diagnosis_flow.py # Based on medical_diagnosis_demo
│   └── test_financial_risk_flow.py # Based on financial_risk_demo
├── demos/
│   ├── __init__.py
│   ├── test_demo_execution.py     # Automated demo testing
│   └── test_master_demo.py        # Full demo suite validation
└── test_full_workflow.py          # End-to-end MCP workflow
```

### 2.3 Unit Test Coverage
**Bayesian Engine Tests:**
- Model creation and validation
- Prior/likelihood/posterior calculations
- MCMC sampling functionality
- Parameter estimation accuracy

**MCP Handler Tests:**
- Request parsing and validation
- Response formatting
- Error handling
- Tool registration

**Schema Tests:**
- Model definition validation
- Parameter constraint checking
- Response serialization

### 2.4 Integration Test Coverage
**Server Integration:**
- FastAPI app startup/shutdown
- Health endpoint functionality
- MCP protocol compliance

**Workflow Tests:**
- Complete A/B testing scenarios (from `ab_testing_demo.py`)
- Medical diagnosis workflows (from `medical_diagnosis_demo.py`)
- Financial risk assessment flows (from `financial_risk_demo.py`)
- Multi-step Bayesian inference chains
- Demo orchestration and automation (from `master_demo.py` and `run_all_demos.py`)

**Demo-Based Tests:**
- Convert each demo into automated test scenarios
- Validate demo outputs match expected statistical results
- Test demo error handling and recovery
- Ensure demos work with renamed module structure

## Phase 3: Implementation Steps

### 3.1 Pre-Implementation Setup
1. **Get pytest best practices** via Context7 MCP server
2. **Analyze existing demo patterns** for test inspiration
3. **Set up test fixtures** for common scenarios

### 3.2 Step-by-Step Execution

#### Step 1: Module Renaming
- [ ] Rename `bayesian_mcp/` to `bayes_mcp/`
- [ ] Rename `bayesian_mcp.py` to `bayes_mcp.py`
- [ ] Update all import statements across codebase
- [ ] Update `pyproject.toml` package configuration
- [ ] Test imports work correctly

#### Step 2: Unit Test Development
- [ ] Create `tests/conftest.py` with shared fixtures
- [ ] Implement `test_bayesian_engine.py`
- [ ] Implement `test_mcp_handlers.py`
- [ ] Implement `test_schemas.py`
- [ ] Implement `test_utils.py`
- [ ] Run unit tests until all pass

#### Step 3: Integration Test Development
- [ ] Create `test_server_lifecycle.py`
- [ ] Convert A/B testing demo to `test_ab_testing_flow.py`
- [ ] Convert medical demo to `test_medical_diagnosis_flow.py`
- [ ] Convert financial demo to `test_financial_risk_flow.py`
- [ ] Create `test_demo_execution.py` (automated demo testing)
- [ ] Create `test_master_demo.py` (full demo suite validation)
- [ ] Create `test_full_workflow.py`
- [ ] Run integration tests until all pass

#### Step 4: Full Test Suite Validation
- [ ] Run complete test suite: `pytest tests/ -v`
- [ ] Fix any failing tests
- [ ] Achieve 100% test success rate
- [ ] Verify test coverage is comprehensive

#### Step 5: Git Operations
- [ ] Stage renamed files and updated imports
- [ ] Commit: "Rename bayesian_mcp to bayes_mcp throughout codebase"
- [ ] Stage new test files
- [ ] Commit: "Add comprehensive unit and integration tests"
- [ ] Push to GitHub repository

## Phase 4: MCP Server Integration Usage

### 4.1 Context7 Integration
- **Purpose:** Get pytest documentation and testing best practices
- **Usage:** Query for advanced pytest patterns, fixtures, async testing
- **Target:** Ensure tests follow industry standards

### 4.2 Sequential Thinking Integration
- **Purpose:** Break down complex testing scenarios
- **Usage:** Analyze multi-step Bayesian workflows for test design
- **Target:** Ensure comprehensive test coverage

### 4.3 GitHub MCP Integration
- **Purpose:** Handle Git operations programmatically
- **Usage:** Automated commit and push operations
- **Target:** Clean commit history and successful push

## Phase 5: Success Criteria

### 5.1 Renaming Success
- [ ] No remaining `bayesian_mcp` references in codebase
- [ ] All imports use `bayes_mcp` correctly
- [ ] Server starts successfully with new module name
- [ ] All demos work with renamed modules

### 5.2 Testing Success
- [ ] Minimum 20 unit tests covering core functionality
- [ ] 5+ integration tests covering workflow scenarios
- [ ] Demo-based tests for all 5 existing demos
- [ ] 100% test pass rate
- [ ] Tests run in under 30 seconds
- [ ] No test dependencies on external services
- [ ] All demos work correctly with renamed modules

### 5.3 Git Success
- [ ] Clean commit history with descriptive messages
- [ ] Successful push to remote repository
- [ ] No merge conflicts
- [ ] All changes properly staged and committed

## Phase 6: Risk Mitigation

### 6.1 Potential Issues
- **Import circular dependencies:** Careful module restructuring
- **Test environment conflicts:** Isolated test fixtures
- **Async test complexity:** Proper pytest-asyncio usage
- **Server port conflicts:** Dynamic port allocation in tests

### 6.2 Backup Strategy
- Create git branch before major changes
- Incremental commits to avoid large failures
- Test each phase independently before proceeding

## Timeline Estimate

- **Phase 1 (Renaming):** 30-45 minutes
- **Phase 2 (Unit Tests):** 90-120 minutes  
- **Phase 3 (Integration Tests):** 60-90 minutes
- **Phase 4 (Git Operations):** 15-30 minutes
- **Total Estimated Time:** 3.5-5 hours

## Next Steps

1. **Review and approve this plan**
2. **Begin Phase 1 execution**
3. **Monitor progress and adapt as needed**
4. **Celebrate successful completion! 🎉**

---

*This plan leverages available MCP servers (Context7, Sequential Thinking, GitHub) to ensure comprehensive, well-tested, and properly versioned code changes.*