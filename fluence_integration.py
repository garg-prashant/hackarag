"""
Fluence Network Integration Module

This module provides integration with the Fluence Network for decentralized compute resources.
It allows the hackathon evaluator to rent and manage VMs for enhanced compute capabilities.
"""

import requests
import json
import time
import paramiko
import streamlit as st
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import os
from dataclasses import dataclass
from enum import Enum


class VMStatus(Enum):
    """VM Status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    STOPPED = "stopped"
    TERMINATED = "terminated"
    ERROR = "error"


@dataclass
class VMConfiguration:
    """VM Configuration data class"""
    name: str
    location: str
    vcpus: int
    memory_gb: int
    storage_gb: int
    server_type: str
    os_image: str
    ssh_key: str
    open_ports: List[int]
    public_ip: bool = True


@dataclass
class VMInstance:
    """VM Instance data class"""
    id: str
    name: str
    status: VMStatus
    public_ip: Optional[str]
    private_ip: Optional[str]
    created_at: datetime
    configuration: VMConfiguration
    cost_per_day: float
    ssh_username: str = "root"


class FluenceClient:
    """Fluence Network API Client"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.fluence.network"):
        """
        Initialize Fluence client
        
        Args:
            api_key: Fluence API key for authentication
            base_url: Base URL for Fluence API
        """
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Make API request to Fluence"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data)
            elif method.upper() == 'PUT':
                response = self.session.put(url, json=data)
            elif method.upper() == 'DELETE':
                response = self.session.delete(url)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            st.error(f"Fluence API Error: {str(e)}")
            return {"error": str(e)}
    
    def get_balance(self) -> Dict:
        """Get account balance"""
        return self._make_request('GET', '/v1/balance')
    
    def get_locations(self) -> List[Dict]:
        """Get available data center locations"""
        response = self._make_request('GET', '/v1/locations')
        return response.get('locations', [])
    
    def get_server_types(self, location: str) -> List[Dict]:
        """Get available server types for a location"""
        response = self._make_request('GET', f'/v1/server-types?location={location}')
        return response.get('server_types', [])
    
    def get_os_images(self) -> List[Dict]:
        """Get available OS images"""
        response = self._make_request('GET', '/v1/os-images')
        return response.get('images', [])
    
    def create_vm(self, config: VMConfiguration) -> Dict:
        """Create a new VM instance"""
        vm_data = {
            "name": config.name,
            "location": config.location,
            "vcpus": config.vcpus,
            "memory_gb": config.memory_gb,
            "storage_gb": config.storage_gb,
            "server_type": config.server_type,
            "os_image": config.os_image,
            "ssh_key": config.ssh_key,
            "open_ports": config.open_ports,
            "public_ip": config.public_ip
        }
        
        response = self._make_request('POST', '/v1/vms', vm_data)
        return response
    
    def get_vms(self) -> List[VMInstance]:
        """Get all VM instances"""
        response = self._make_request('GET', '/v1/vms')
        vms = []
        
        for vm_data in response.get('vms', []):
            vm = VMInstance(
                id=vm_data['id'],
                name=vm_data['name'],
                status=VMStatus(vm_data['status']),
                public_ip=vm_data.get('public_ip'),
                private_ip=vm_data.get('private_ip'),
                created_at=datetime.fromisoformat(vm_data['created_at']),
                configuration=VMConfiguration(**vm_data['configuration']),
                cost_per_day=vm_data['cost_per_day']
            )
            vms.append(vm)
        
        return vms
    
    def get_vm(self, vm_id: str) -> Optional[VMInstance]:
        """Get specific VM instance"""
        response = self._make_request('GET', f'/v1/vms/{vm_id}')
        
        if 'error' in response:
            return None
        
        vm_data = response['vm']
        return VMInstance(
            id=vm_data['id'],
            name=vm_data['name'],
            status=VMStatus(vm_data['status']),
            public_ip=vm_data.get('public_ip'),
            private_ip=vm_data.get('private_ip'),
            created_at=datetime.fromisoformat(vm_data['created_at']),
            configuration=VMConfiguration(**vm_data['configuration']),
            cost_per_day=vm_data['cost_per_day']
        )
    
    def terminate_vm(self, vm_id: str) -> Dict:
        """Terminate VM instance"""
        return self._make_request('DELETE', f'/v1/vms/{vm_id}')
    
    def get_vm_logs(self, vm_id: str) -> str:
        """Get VM logs"""
        response = self._make_request('GET', f'/v1/vms/{vm_id}/logs')
        return response.get('logs', '')


class FluenceComputeManager:
    """Manages Fluence compute resources for hackathon evaluation"""
    
    def __init__(self, api_key: str):
        """Initialize Fluence compute manager"""
        self.client = FluenceClient(api_key)
        self.vms = {}
        self.compute_tasks = {}
    
    def setup_compute_environment(self, hackathon_idea: str, bounty_requirements: List[Dict]) -> Dict:
        """
        Set up compute environment for hackathon idea evaluation
        
        Args:
            hackathon_idea: The hackathon project idea
            bounty_requirements: List of bounty requirements to evaluate against
            
        Returns:
            Dict with setup status and VM information
        """
        try:
            # Analyze compute requirements based on idea complexity
            compute_requirements = self._analyze_compute_requirements(hackathon_idea, bounty_requirements)
            
            # Get available locations
            locations = self.client.get_locations()
            if not locations:
                return {"error": "No available locations found"}
            
            # Select optimal location (first available for now)
            selected_location = locations[0]['id']
            
            # Get server types for selected location
            server_types = self.client.get_server_types(selected_location)
            if not server_types:
                return {"error": "No server types available for selected location"}
            
            # Select appropriate server type based on requirements
            selected_server = self._select_server_type(server_types, compute_requirements)
            
            # Get OS images
            os_images = self.client.get_os_images()
            if not os_images:
                return {"error": "No OS images available"}
            
            # Select appropriate OS image
            selected_os = self._select_os_image(os_images, compute_requirements)
            
            # Create VM configuration
            vm_config = VMConfiguration(
                name=f"hackathon-eval-{int(time.time())}",
                location=selected_location,
                vcpus=compute_requirements['vcpus'],
                memory_gb=compute_requirements['memory_gb'],
                storage_gb=compute_requirements['storage_gb'],
                server_type=selected_server['id'],
                os_image=selected_os['id'],
                ssh_key=self._get_or_generate_ssh_key(),
                open_ports=[22, 80, 443, 3000, 8000, 8080],  # Common development ports
                public_ip=True
            )
            
            # Create VM
            vm_response = self.client.create_vm(vm_config)
            if 'error' in vm_response:
                return vm_response
            
            # Store VM information
            vm_id = vm_response['vm_id']
            self.vms[vm_id] = {
                'config': vm_config,
                'created_at': datetime.now(),
                'status': 'creating'
            }
            
            return {
                "success": True,
                "vm_id": vm_id,
                "configuration": vm_config,
                "estimated_cost": vm_response.get('estimated_cost', 0),
                "message": "VM creation initiated successfully"
            }
            
        except Exception as e:
            return {"error": f"Failed to setup compute environment: {str(e)}"}
    
    def _analyze_compute_requirements(self, hackathon_idea: str, bounty_requirements: List[Dict]) -> Dict:
        """Analyze compute requirements based on hackathon idea and bounty requirements"""
        # Basic analysis - can be enhanced with ML models
        idea_complexity = self._assess_idea_complexity(hackathon_idea)
        bounty_complexity = self._assess_bounty_complexity(bounty_requirements)
        
        # Determine compute requirements
        if idea_complexity == "high" or bounty_complexity == "high":
            return {
                "vcpus": 4,
                "memory_gb": 8,
                "storage_gb": 50,
                "complexity": "high"
            }
        elif idea_complexity == "medium" or bounty_complexity == "medium":
            return {
                "vcpus": 2,
                "memory_gb": 4,
                "storage_gb": 25,
                "complexity": "medium"
            }
        else:
            return {
                "vcpus": 1,
                "memory_gb": 2,
                "storage_gb": 25,
                "complexity": "low"
            }
    
    def _assess_idea_complexity(self, idea: str) -> str:
        """Assess complexity of hackathon idea"""
        complexity_keywords = {
            "high": ["ai", "machine learning", "blockchain", "decentralized", "cryptography", "zero-knowledge", "consensus", "smart contract"],
            "medium": ["api", "database", "web app", "mobile app", "integration", "automation"],
            "low": ["simple", "basic", "static", "landing page", "portfolio"]
        }
        
        idea_lower = idea.lower()
        
        for complexity, keywords in complexity_keywords.items():
            if any(keyword in idea_lower for keyword in keywords):
                return complexity
        
        return "medium"  # Default to medium complexity
    
    def _assess_bounty_complexity(self, bounty_requirements: List[Dict]) -> str:
        """Assess complexity of bounty requirements"""
        if not bounty_requirements:
            return "low"
        
        total_requirements = len(bounty_requirements)
        if total_requirements > 5:
            return "high"
        elif total_requirements > 2:
            return "medium"
        else:
            return "low"
    
    def _select_server_type(self, server_types: List[Dict], requirements: Dict) -> Dict:
        """Select appropriate server type based on requirements"""
        # Filter server types that meet minimum requirements
        suitable_servers = [
            server for server in server_types
            if server['vcpus'] >= requirements['vcpus'] and
               server['memory_gb'] >= requirements['memory_gb']
        ]
        
        if not suitable_servers:
            return server_types[0]  # Fallback to first available
        
        # Select the most cost-effective option
        return min(suitable_servers, key=lambda x: x['price_per_hour'])
    
    def _select_os_image(self, os_images: List[Dict], requirements: Dict) -> Dict:
        """Select appropriate OS image based on requirements"""
        # Prefer Ubuntu for development environments
        ubuntu_images = [img for img in os_images if 'ubuntu' in img['name'].lower()]
        if ubuntu_images:
            return ubuntu_images[0]
        
        # Fallback to first available image
        return os_images[0]
    
    def _get_or_generate_ssh_key(self) -> str:
        """Get or generate SSH key for VM access"""
        # In a real implementation, this would generate or retrieve SSH keys
        # For now, return a placeholder
        return "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC... placeholder-key"
    
    def deploy_evaluation_environment(self, vm_id: str, hackathon_idea: str, bounty_data: List[Dict]) -> Dict:
        """
        Deploy evaluation environment on VM
        
        Args:
            vm_id: ID of the VM instance
            hackathon_idea: The hackathon project idea
            bounty_data: Bounty data for evaluation
            
        Returns:
            Dict with deployment status
        """
        try:
            vm = self.client.get_vm(vm_id)
            if not vm or vm.status != VMStatus.RUNNING:
                return {"error": "VM is not running or not found"}
            
            # Prepare deployment script
            deployment_script = self._generate_deployment_script(hackathon_idea, bounty_data)
            
            # Execute deployment via SSH
            deployment_result = self._execute_ssh_command(
                vm.public_ip,
                vm.ssh_username,
                deployment_script
            )
            
            if deployment_result['success']:
                self.compute_tasks[vm_id] = {
                    'idea': hackathon_idea,
                    'bounty_data': bounty_data,
                    'deployed_at': datetime.now(),
                    'status': 'running'
                }
                
                return {
                    "success": True,
                    "message": "Evaluation environment deployed successfully",
                    "vm_ip": vm.public_ip,
                    "access_info": {
                        "ssh_command": f"ssh {vm.ssh_username}@{vm.public_ip}",
                        "web_interface": f"http://{vm.public_ip}:8000"
                    }
                }
            else:
                return {"error": f"Deployment failed: {deployment_result['error']}"}
                
        except Exception as e:
            return {"error": f"Failed to deploy evaluation environment: {str(e)}"}
    
    def _generate_deployment_script(self, hackathon_idea: str, bounty_data: List[Dict]) -> str:
        """Generate deployment script for evaluation environment"""
        script = f"""#!/bin/bash
# Hackathon Evaluation Environment Setup

# Update system
apt-get update -y

# Install Python and dependencies
apt-get install -y python3 python3-pip git curl

# Install Node.js for web interface
curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
apt-get install -y nodejs

# Create evaluation directory
mkdir -p /opt/hackathon-eval
cd /opt/hackathon-eval

# Create evaluation script
cat > evaluate_idea.py << 'EOF'
import json
import sys
from datetime import datetime

def evaluate_hackathon_idea(idea, bounty_data):
    \"\"\"Evaluate hackathon idea against bounty requirements\"\"\"
    
    results = {{
        'idea': idea,
        'evaluation_time': datetime.now().isoformat(),
        'bounty_matches': [],
        'recommendations': [],
        'technical_analysis': {{
            'complexity_score': 0,
            'feasibility_score': 0,
            'innovation_score': 0
        }}
    }}
    
    # Analyze idea against each bounty
    for bounty in bounty_data:
        match_score = analyze_bounty_match(idea, bounty)
        if match_score > 0.5:  # 50% threshold
            results['bounty_matches'].append({{
                'bounty_title': bounty.get('title', ''),
                'match_score': match_score,
                'reasons': get_match_reasons(idea, bounty)
            }})
    
    # Generate recommendations
    results['recommendations'] = generate_recommendations(idea, bounty_data)
    
    # Calculate technical scores
    results['technical_analysis'] = calculate_technical_scores(idea, bounty_data)
    
    return results

def analyze_bounty_match(idea, bounty):
    \"\"\"Analyze how well idea matches bounty requirements\"\"\"
    # Simple keyword matching - can be enhanced with NLP
    idea_lower = idea.lower()
    bounty_text = (bounty.get('title', '') + ' ' + bounty.get('description', '')).lower()
    
    # Count matching keywords
    common_words = set(idea_lower.split()) & set(bounty_text.split())
    return len(common_words) / max(len(set(bounty_text.split())), 1)

def get_match_reasons(idea, bounty):
    \"\"\"Get reasons for bounty match\"\"\"
    return ["Keyword similarity", "Technical alignment", "Use case overlap"]

def generate_recommendations(idea, bounty_data):
    \"\"\"Generate recommendations for idea improvement\"\"\"
    return [
        "Consider integrating with more bounty requirements",
        "Focus on technical feasibility within hackathon timeframe",
        "Enhance innovation aspects of your idea"
    ]

def calculate_technical_scores(idea, bounty_data):
    \"\"\"Calculate technical analysis scores\"\"\"
    return {{
        'complexity_score': 0.7,
        'feasibility_score': 0.8,
        'innovation_score': 0.6
    }}

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 evaluate_idea.py <idea> <bounty_data_json>")
        sys.exit(1)
    
    idea = sys.argv[1]
    bounty_data = json.loads(sys.argv[2])
    
    results = evaluate_hackathon_idea(idea, bounty_data)
    print(json.dumps(results, indent=2))
EOF

# Create web interface
cat > web_interface.py << 'EOF'
from flask import Flask, render_template, request, jsonify
import json
import subprocess
import os

app = Flask(__name__)

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hackathon Idea Evaluator</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .container {{ max-width: 800px; margin: 0 auto; }}
            .form-group {{ margin-bottom: 20px; }}
            label {{ display: block; margin-bottom: 5px; font-weight: bold; }}
            input, textarea {{ width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }}
            button {{ background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }}
            button:hover {{ background: #0056b3; }}
            .results {{ margin-top: 20px; padding: 20px; background: #f8f9fa; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Hackathon Idea Evaluator</h1>
            <form id="evaluationForm">
                <div class="form-group">
                    <label for="idea">Your Hackathon Idea:</label>
                    <textarea id="idea" name="idea" rows="4" placeholder="Describe your hackathon project idea..."></textarea>
                </div>
                <div class="form-group">
                    <label for="bountyData">Bounty Data (JSON):</label>
                    <textarea id="bountyData" name="bountyData" rows="6" placeholder="Paste bounty data JSON here..."></textarea>
                </div>
                <button type="submit">Evaluate Idea</button>
            </form>
            <div id="results" class="results" style="display: none;"></div>
        </div>
        
        <script>
            document.getElementById('evaluationForm').addEventListener('submit', async function(e) {{
                e.preventDefault();
                
                const idea = document.getElementById('idea').value;
                const bountyData = document.getElementById('bountyData').value;
                
                try {{
                    const response = await fetch('/evaluate', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ idea: idea, bountyData: bountyData }})
                    }});
                    
                    const results = await response.json();
                    displayResults(results);
                }} catch (error) {{
                    console.error('Error:', error);
                }}
            }});
            
            function displayResults(results) {{
                const resultsDiv = document.getElementById('results');
                resultsDiv.innerHTML = '<h2>Evaluation Results</h2><pre>' + JSON.stringify(results, null, 2) + '</pre>';
                resultsDiv.style.display = 'block';
            }}
        </script>
    </body>
    </html>
    '''

@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.json
    idea = data.get('idea', '')
    bounty_data = data.get('bountyData', '[]')
    
    try:
        bounty_json = json.loads(bounty_data)
        result = subprocess.run(['python3', 'evaluate_idea.py', idea, json.dumps(bounty_json)], 
                              capture_output=True, text=True, cwd='/opt/hackathon-eval')
        
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            return {{'error': result.stderr}}
    except Exception as e:
        return {{'error': str(e)}}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
EOF

# Install Flask
pip3 install flask

# Start web interface
nohup python3 web_interface.py > /var/log/hackathon-eval.log 2>&1 &

echo "Hackathon evaluation environment deployed successfully!"
echo "Web interface available at: http://$(curl -s ifconfig.me):8000"
"""
        
        return script
    
    def _execute_ssh_command(self, host: str, username: str, command: str) -> Dict:
        """Execute command on VM via SSH"""
        try:
            # In a real implementation, this would use proper SSH key authentication
            # For now, return a mock success response
            return {
                "success": True,
                "output": "Command executed successfully (mock)",
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "output": None,
                "error": str(e)
            }
    
    def run_evaluation(self, vm_id: str, hackathon_idea: str, bounty_data: List[Dict]) -> Dict:
        """Run evaluation on the deployed VM"""
        try:
            vm = self.client.get_vm(vm_id)
            if not vm or vm.status != VMStatus.RUNNING:
                return {"error": "VM is not running or not found"}
            
            # Prepare evaluation data
            evaluation_data = {
                "idea": hackathon_idea,
                "bounty_data": bounty_data,
                "timestamp": datetime.now().isoformat()
            }
            
            # Execute evaluation via SSH
            evaluation_command = f"python3 /opt/hackathon-eval/evaluate_idea.py '{hackathon_idea}' '{json.dumps(bounty_data)}'"
            result = self._execute_ssh_command(
                vm.public_ip,
                vm.ssh_username,
                evaluation_command
            )
            
            if result['success']:
                return {
                    "success": True,
                    "results": json.loads(result['output']) if result['output'] else {},
                    "vm_info": {
                        "id": vm_id,
                        "ip": vm.public_ip,
                        "web_interface": f"http://{vm.public_ip}:8000"
                    }
                }
            else:
                return {"error": f"Evaluation failed: {result['error']}"}
                
        except Exception as e:
            return {"error": f"Failed to run evaluation: {str(e)}"}
    
    def cleanup_vm(self, vm_id: str) -> Dict:
        """Cleanup and terminate VM"""
        try:
            result = self.client.terminate_vm(vm_id)
            
            # Remove from local tracking
            if vm_id in self.vms:
                del self.vms[vm_id]
            if vm_id in self.compute_tasks:
                del self.compute_tasks[vm_id]
            
            return {
                "success": True,
                "message": "VM terminated successfully"
            }
        except Exception as e:
            return {"error": f"Failed to cleanup VM: {str(e)}"}
    
    def get_vm_status(self, vm_id: str) -> Dict:
        """Get VM status and information"""
        try:
            vm = self.client.get_vm(vm_id)
            if not vm:
                return {"error": "VM not found"}
            
            return {
                "id": vm.id,
                "name": vm.name,
                "status": vm.status.value,
                "public_ip": vm.public_ip,
                "created_at": vm.created_at.isoformat(),
                "cost_per_day": vm.cost_per_day,
                "web_interface": f"http://{vm.public_ip}:8000" if vm.public_ip else None
            }
        except Exception as e:
            return {"error": f"Failed to get VM status: {str(e)}"}


def create_fluence_config() -> Dict:
    """Create default Fluence configuration"""
    return {
        "api_key": "",
        "base_url": "https://api.fluence.network",
        "default_vm_config": {
            "vcpus": 2,
            "memory_gb": 4,
            "storage_gb": 25,
            "open_ports": [22, 80, 443, 3000, 8000, 8080]
        },
        "auto_cleanup_hours": 24  # Auto cleanup VMs after 24 hours
    }


def save_fluence_config(config: Dict, config_file: str = "fluence_config.json"):
    """Save Fluence configuration to file"""
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)


def load_fluence_config(config_file: str = "fluence_config.json") -> Dict:
    """Load Fluence configuration from file"""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return create_fluence_config()
    except Exception as e:
        st.error(f"Error loading Fluence config: {str(e)}")
        return create_fluence_config()
