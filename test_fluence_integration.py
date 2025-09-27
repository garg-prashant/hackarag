"""
Test script for Fluence Network integration

This script tests the basic functionality of the Fluence integration
without requiring actual API credentials.
"""

import json
import sys
import os
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fluence_integration import (
    FluenceClient, 
    FluenceComputeManager, 
    VMConfiguration, 
    VMInstance, 
    VMStatus,
    create_fluence_config,
    save_fluence_config,
    load_fluence_config
)


def test_fluence_config():
    """Test Fluence configuration management"""
    print("🧪 Testing Fluence configuration...")
    
    # Test creating default config
    config = create_fluence_config()
    assert 'api_key' in config
    assert 'base_url' in config
    assert 'default_vm_config' in config
    print("✅ Configuration creation works")
    
    # Test saving and loading config
    test_config_file = "test_fluence_config.json"
    save_fluence_config(config, test_config_file)
    
    loaded_config = load_fluence_config(test_config_file)
    assert loaded_config == config
    print("✅ Configuration save/load works")
    
    # Cleanup
    if os.path.exists(test_config_file):
        os.remove(test_config_file)
    
    print("✅ Configuration tests passed\n")


def test_vm_configuration():
    """Test VM configuration data class"""
    print("🧪 Testing VM configuration...")
    
    config = VMConfiguration(
        name="test-vm",
        location="us-east-1",
        vcpus=2,
        memory_gb=4,
        storage_gb=25,
        server_type="standard",
        os_image="ubuntu-20.04",
        ssh_key="ssh-rsa test-key",
        open_ports=[22, 80, 443]
    )
    
    assert config.name == "test-vm"
    assert config.vcpus == 2
    assert config.memory_gb == 4
    assert config.storage_gb == 25
    assert len(config.open_ports) == 3
    print("✅ VM configuration works")
    
    print("✅ VM configuration tests passed\n")


def test_vm_instance():
    """Test VM instance data class"""
    print("🧪 Testing VM instance...")
    
    config = VMConfiguration(
        name="test-vm",
        location="us-east-1",
        vcpus=2,
        memory_gb=4,
        storage_gb=25,
        server_type="standard",
        os_image="ubuntu-20.04",
        ssh_key="ssh-rsa test-key",
        open_ports=[22, 80, 443]
    )
    
    vm = VMInstance(
        id="vm-123",
        name="test-vm",
        status=VMStatus.RUNNING,
        public_ip="192.168.1.100",
        private_ip="10.0.0.100",
        created_at=datetime.now(),
        configuration=config,
        cost_per_day=5.0
    )
    
    assert vm.id == "vm-123"
    assert vm.status == VMStatus.RUNNING
    assert vm.public_ip == "192.168.1.100"
    assert vm.cost_per_day == 5.0
    print("✅ VM instance works")
    
    print("✅ VM instance tests passed\n")


def test_fluence_client_mock():
    """Test Fluence client with mock responses"""
    print("🧪 Testing Fluence client (mock)...")
    
    # Test with empty API key (should fail gracefully)
    client = FluenceClient("", "https://api.fluence.network")
    
    # Test methods that would make API calls
    # These will fail but should not crash
    balance = client.get_balance()
    assert 'error' in balance
    print("✅ Client handles API errors gracefully")
    
    locations = client.get_locations()
    assert 'error' in locations
    print("✅ Client handles missing API key")
    
    print("✅ Fluence client tests passed\n")


def test_compute_manager_mock():
    """Test compute manager with mock data"""
    print("🧪 Testing compute manager (mock)...")
    
    # Test with empty API key
    manager = FluenceComputeManager("")
    
    # Test compute requirements analysis
    idea = "I'm building a decentralized voting system using blockchain"
    bounty_requirements = [
        {"title": "Blockchain Integration", "description": "Use blockchain technology"},
        {"title": "Voting System", "description": "Implement secure voting"}
    ]
    
    requirements = manager._analyze_compute_requirements(idea, bounty_requirements)
    assert 'vcpus' in requirements
    assert 'memory_gb' in requirements
    assert 'storage_gb' in requirements
    assert 'complexity' in requirements
    print("✅ Compute requirements analysis works")
    
    # Test idea complexity assessment
    complexity = manager._assess_idea_complexity("Simple web app")
    assert complexity in ["low", "medium", "high"]
    print("✅ Idea complexity assessment works")
    
    # Test bounty complexity assessment
    complexity = manager._assess_bounty_complexity(bounty_requirements)
    assert complexity in ["low", "medium", "high"]
    print("✅ Bounty complexity assessment works")
    
    print("✅ Compute manager tests passed\n")


def test_deployment_script_generation():
    """Test deployment script generation"""
    print("🧪 Testing deployment script generation...")
    
    manager = FluenceComputeManager("")
    
    idea = "Test hackathon idea"
    bounty_data = [{"title": "Test Bounty", "description": "Test description"}]
    
    script = manager._generate_deployment_script(idea, bounty_data)
    
    assert isinstance(script, str)
    assert "#!/bin/bash" in script
    assert "python3" in script
    assert "flask" in script
    assert "evaluate_idea.py" in script
    print("✅ Deployment script generation works")
    
    print("✅ Deployment script tests passed\n")


def main():
    """Run all tests"""
    print("🚀 Starting Fluence Integration Tests\n")
    
    try:
        test_fluence_config()
        test_vm_configuration()
        test_vm_instance()
        test_fluence_client_mock()
        test_compute_manager_mock()
        test_deployment_script_generation()
        
        print("🎉 All tests passed! Fluence integration is working correctly.")
        print("\n📝 Note: These tests use mock data. For full functionality, configure a valid Fluence API key.")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
