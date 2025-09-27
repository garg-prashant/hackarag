"""
Fluence Network UI Components for Streamlit

This module provides UI components for integrating Fluence Network capabilities
into the hackathon evaluator application.
"""

import streamlit as st
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from fluence_integration import FluenceComputeManager, create_fluence_config, load_fluence_config, save_fluence_config


def render_fluence_sidebar():
    """Render Fluence integration sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🌐 Fluence Network")
        
        # Check if Fluence is configured
        config = load_fluence_config()
        if not config.get('api_key'):
            st.warning("⚠️ Fluence API key not configured")
            if st.button("🔧 Configure Fluence", key="configure_fluence_sidebar"):
                st.session_state.show_fluence_config = True
        else:
            st.success("✅ Fluence configured")
            
            # Show active VMs
            if 'fluence_manager' in st.session_state:
                manager = st.session_state.fluence_manager
                vms = manager.client.get_vms()
                
                if vms:
                    st.markdown("#### 🖥️ Active VMs")
                    for vm in vms:
                        status_color = {
                            'running': '🟢',
                            'pending': '🟡',
                            'stopped': '🔴',
                            'terminated': '⚫',
                            'error': '❌'
                        }.get(vm.status.value, '⚪')
                        
                        with st.expander(f"{status_color} {vm.name}"):
                            st.write(f"**Status:** {vm.status.value}")
                            st.write(f"**IP:** {vm.public_ip or 'N/A'}")
                            st.write(f"**Created:** {vm.created_at.strftime('%Y-%m-%d %H:%M')}")
                            st.write(f"**Cost/Day:** ${vm.cost_per_day:.2f}")
                            
                            if vm.public_ip and vm.status.value == 'running':
                                st.write(f"**Web Interface:** [Open](http://{vm.public_ip}:8000)")
                            
                            if st.button(f"🗑️ Terminate", key=f"terminate_{vm.id}"):
                                result = manager.cleanup_vm(vm.id)
                                if result.get('success'):
                                    st.success("VM terminated successfully")
                                    st.rerun()
                                else:
                                    st.error(f"Failed to terminate VM: {result.get('error')}")
                else:
                    st.info("No active VMs")
            
            if st.button("🔄 Refresh VMs", key="refresh_vms_sidebar"):
                st.rerun()


def render_fluence_config_modal():
    """Render Fluence configuration modal"""
    if st.session_state.get('show_fluence_config', False):
        with st.expander("🔧 Configure Fluence Network", expanded=True):
            st.markdown("### Fluence Network Configuration")
            st.markdown("Configure your Fluence Network API access for decentralized compute resources.")
            
            # Load current config
            config = load_fluence_config()
            
            # API Key input
            api_key = st.text_input(
                "API Key",
                value=config.get('api_key', ''),
                type="password",
                help="Your Fluence Network API key. Get it from the Fluence Console."
            )
            
            # Base URL
            base_url = st.text_input(
                "Base URL",
                value=config.get('base_url', 'https://api.fluence.network'),
                help="Fluence API base URL"
            )
            
            # Default VM Configuration
            st.markdown("#### Default VM Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                default_vcpus = st.number_input(
                    "vCPUs",
                    min_value=1,
                    max_value=8,
                    value=config.get('default_vm_config', {}).get('vcpus', 2),
                    help="Number of virtual CPUs"
                )
                
                default_memory = st.number_input(
                    "Memory (GB)",
                    min_value=1,
                    max_value=32,
                    value=config.get('default_vm_config', {}).get('memory_gb', 4),
                    help="Memory in GB"
                )
            
            with col2:
                default_storage = st.number_input(
                    "Storage (GB)",
                    min_value=25,
                    max_value=1000,
                    value=config.get('default_vm_config', {}).get('storage_gb', 25),
                    help="Storage in GB"
                )
                
                auto_cleanup = st.number_input(
                    "Auto Cleanup (hours)",
                    min_value=1,
                    max_value=168,
                    value=config.get('auto_cleanup_hours', 24),
                    help="Automatically cleanup VMs after this many hours"
                )
            
            # Open Ports
            st.markdown("#### Open Ports")
            open_ports_input = st.text_input(
                "Open Ports (comma-separated)",
                value=",".join(map(str, config.get('default_vm_config', {}).get('open_ports', [22, 80, 443, 3000, 8000, 8080]))),
                help="Ports to open on the VM (comma-separated)"
            )
            
            try:
                open_ports = [int(port.strip()) for port in open_ports_input.split(',') if port.strip()]
            except ValueError:
                open_ports = [22, 80, 443, 3000, 8000, 8080]
                st.error("Invalid port format. Using default ports.")
            
            # Save configuration
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if st.button("💾 Save", key="save_fluence_config"):
                    new_config = {
                        "api_key": api_key,
                        "base_url": base_url,
                        "default_vm_config": {
                            "vcpus": default_vcpus,
                            "memory_gb": default_memory,
                            "storage_gb": default_storage,
                            "open_ports": open_ports
                        },
                        "auto_cleanup_hours": auto_cleanup
                    }
                    
                    save_fluence_config(new_config)
                    st.success("Configuration saved successfully!")
                    st.session_state.show_fluence_config = False
                    st.rerun()
            
            with col2:
                if st.button("❌ Cancel", key="cancel_fluence_config"):
                    st.session_state.show_fluence_config = False
                    st.rerun()
            
            with col3:
                if st.button("🧪 Test Connection", key="test_fluence_connection"):
                    if api_key:
                        try:
                            from fluence_integration import FluenceClient
                            client = FluenceClient(api_key, base_url)
                            balance = client.get_balance()
                            
                            if 'error' not in balance:
                                st.success("✅ Connection successful!")
                                st.json(balance)
                            else:
                                st.error(f"❌ Connection failed: {balance['error']}")
                        except Exception as e:
                            st.error(f"❌ Connection failed: {str(e)}")
                    else:
                        st.warning("Please enter an API key first")


def render_fluence_compute_section():
    """Render Fluence compute capabilities section"""
    st.markdown("---")
    st.markdown("### 🌐 Fluence Network Compute")
    st.markdown("Leverage decentralized compute resources for enhanced hackathon idea evaluation.")
    
    # Check if Fluence is configured
    config = load_fluence_config()
    if not config.get('api_key'):
        st.warning("⚠️ Fluence Network not configured. Please configure it in the sidebar to use compute features.")
        return
    
    # Initialize Fluence manager if not exists
    if 'fluence_manager' not in st.session_state:
        st.session_state.fluence_manager = FluenceComputeManager(config['api_key'])
    
    manager = st.session_state.fluence_manager
    
    # Show current VMs
    vms = manager.client.get_vms()
    
    if vms:
        st.markdown("#### 🖥️ Active Compute Resources")
        
        for vm in vms:
            with st.container():
                col1, col2, col3 = st.columns([3, 2, 1])
                
                with col1:
                    status_icon = {
                        'running': '🟢',
                        'pending': '🟡',
                        'stopped': '🔴',
                        'terminated': '⚫',
                        'error': '❌'
                    }.get(vm.status.value, '⚪')
                    
                    st.markdown(f"**{status_icon} {vm.name}**")
                    st.caption(f"Status: {vm.status.value} | IP: {vm.public_ip or 'N/A'}")
                
                with col2:
                    st.metric("Cost/Day", f"${vm.cost_per_day:.2f}")
                    st.caption(f"Created: {vm.created_at.strftime('%m/%d %H:%M')}")
                
                with col3:
                    if vm.status.value == 'running' and vm.public_ip:
                        st.link_button("🌐 Open", f"http://{vm.public_ip}:8000")
                    
                    if st.button("🗑️", key=f"terminate_vm_{vm.id}", help="Terminate VM"):
                        result = manager.cleanup_vm(vm.id)
                        if result.get('success'):
                            st.success("VM terminated successfully")
                            st.rerun()
                        else:
                            st.error(f"Failed to terminate VM: {result.get('error')}")
    
    # Compute options
    st.markdown("#### 🚀 Compute Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🆕 Deploy New VM", key="deploy_new_vm", use_container_width=True):
            if st.session_state.get('user_idea') and st.session_state.get('selected_bounties'):
                with st.spinner("Setting up compute environment..."):
                    # Get bounty data for selected bounties
                    bounty_data = []
                    for bounty_id in st.session_state.selected_bounties:
                        bounty_info = st.session_state.vectorizer.get_bounty_by_id(bounty_id)
                        if bounty_info:
                            bounty_data.append({
                                'title': bounty_info['metadata'].get('title', ''),
                                'description': bounty_info['document'],
                                'company': bounty_info['metadata'].get('company', '')
                            })
                    
                    # Setup compute environment
                    result = manager.setup_compute_environment(
                        st.session_state.user_idea,
                        bounty_data
                    )
                    
                    if result.get('success'):
                        st.success("✅ Compute environment setup initiated!")
                        st.json(result)
                        st.rerun()
                    else:
                        st.error(f"❌ Setup failed: {result.get('error')}")
            else:
                st.warning("Please complete the idea evaluation steps first")
    
    with col2:
        if st.button("🔄 Refresh Status", key="refresh_vm_status", use_container_width=True):
            st.rerun()
    
    # Advanced compute features
    if vms and any(vm.status.value == 'running' for vm in vms):
        st.markdown("#### ⚡ Advanced Compute Features")
        
        running_vms = [vm for vm in vms if vm.status.value == 'running']
        
        if running_vms:
            selected_vm = st.selectbox(
                "Select VM for advanced operations:",
                options=[(vm.id, vm.name) for vm in running_vms],
                format_func=lambda x: x[1]
            )
            
            if selected_vm:
                vm_id = selected_vm[0]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🚀 Deploy Evaluation", key="deploy_evaluation"):
                        if st.session_state.get('user_idea') and st.session_state.get('selected_bounties'):
                            with st.spinner("Deploying evaluation environment..."):
                                bounty_data = []
                                for bounty_id in st.session_state.selected_bounties:
                                    bounty_info = st.session_state.vectorizer.get_bounty_by_id(bounty_id)
                                    if bounty_info:
                                        bounty_data.append({
                                            'title': bounty_info['metadata'].get('title', ''),
                                            'description': bounty_info['document'],
                                            'company': bounty_info['metadata'].get('company', '')
                                        })
                                
                                result = manager.deploy_evaluation_environment(
                                    vm_id,
                                    st.session_state.user_idea,
                                    bounty_data
                                )
                                
                                if result.get('success'):
                                    st.success("✅ Evaluation environment deployed!")
                                    st.json(result)
                                else:
                                    st.error(f"❌ Deployment failed: {result.get('error')}")
                        else:
                            st.warning("Please complete the idea evaluation steps first")
                
                with col2:
                    if st.button("🔍 Run Analysis", key="run_analysis"):
                        if st.session_state.get('user_idea') and st.session_state.get('selected_bounties'):
                            with st.spinner("Running advanced analysis..."):
                                bounty_data = []
                                for bounty_id in st.session_state.selected_bounties:
                                    bounty_info = st.session_state.vectorizer.get_bounty_by_id(bounty_id)
                                    if bounty_info:
                                        bounty_data.append({
                                            'title': bounty_info['metadata'].get('title', ''),
                                            'description': bounty_info['document'],
                                            'company': bounty_info['metadata'].get('company', '')
                                        })
                                
                                result = manager.run_evaluation(
                                    vm_id,
                                    st.session_state.user_idea,
                                    bounty_data
                                )
                                
                                if result.get('success'):
                                    st.success("✅ Analysis completed!")
                                    
                                    # Display results
                                    if 'results' in result:
                                        st.markdown("#### 📊 Analysis Results")
                                        st.json(result['results'])
                                    
                                    # Show VM info
                                    if 'vm_info' in result:
                                        vm_info = result['vm_info']
                                        st.markdown("#### 🖥️ VM Information")
                                        st.write(f"**Web Interface:** [Open]({vm_info['web_interface']})")
                                else:
                                    st.error(f"❌ Analysis failed: {result.get('error')}")
                        else:
                            st.warning("Please complete the idea evaluation steps first")
                
                with col3:
                    if st.button("📊 Get Status", key="get_vm_status"):
                        with st.spinner("Getting VM status..."):
                            result = manager.get_vm_status(vm_id)
                            
                            if 'error' not in result:
                                st.success("✅ Status retrieved!")
                                st.json(result)
                            else:
                                st.error(f"❌ Failed to get status: {result['error']}")


def render_fluence_metrics():
    """Render Fluence usage metrics"""
    if 'fluence_manager' in st.session_state:
        manager = st.session_state.fluence_manager
        
        try:
            # Get account balance
            balance = manager.client.get_balance()
            
            if 'error' not in balance:
                st.markdown("#### 💰 Account Balance")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Available Balance", f"${balance.get('available', 0):.2f}")
                
                with col2:
                    st.metric("Reserved Balance", f"${balance.get('reserved', 0):.2f}")
                
                with col3:
                    st.metric("Total Balance", f"${balance.get('total', 0):.2f}")
            
            # Get VM statistics
            vms = manager.client.get_vms()
            
            if vms:
                st.markdown("#### 📈 Usage Statistics")
                
                total_cost = sum(vm.cost_per_day for vm in vms)
                running_vms = len([vm for vm in vms if vm.status.value == 'running'])
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total VMs", len(vms))
                
                with col2:
                    st.metric("Running VMs", running_vms)
                
                with col3:
                    st.metric("Daily Cost", f"${total_cost:.2f}")
        
        except Exception as e:
            st.error(f"Error loading metrics: {str(e)}")


def render_fluence_help():
    """Render Fluence help section"""
    with st.expander("❓ Fluence Network Help", expanded=False):
        st.markdown("""
        ### What is Fluence Network?
        
        Fluence Network is a decentralized compute marketplace that provides access to enterprise-grade 
        compute resources from data centers around the world. It allows you to rent virtual machines 
        for enhanced hackathon idea evaluation and testing.
        
        ### Key Features
        
        - **Decentralized Compute**: Access compute resources from multiple providers
        - **Pay-per-use**: Only pay for what you use
        - **Global Infrastructure**: Data centers worldwide for low latency
        - **Flexible Configuration**: Choose CPU, memory, and storage as needed
        - **Easy Management**: Simple web interface and API access
        
        ### How to Get Started
        
        1. **Get API Key**: Sign up at [Fluence Console](https://console.fluence.network)
        2. **Configure**: Enter your API key in the configuration section
        3. **Deploy**: Create VMs for your hackathon projects
        4. **Evaluate**: Use enhanced compute for idea analysis
        
        ### Use Cases for Hackathons
        
        - **Heavy Computation**: Run complex algorithms and analysis
        - **Testing Environments**: Deploy and test your applications
        - **AI/ML Processing**: Train models and run inference
        - **Blockchain Development**: Test smart contracts and dApps
        - **Collaborative Development**: Share compute resources with team
        
        ### Cost Management
        
        - VMs are charged daily at 5:55 PM UTC
        - Prepayment covers the next day's rent
        - Terminate VMs when not needed to save costs
        - Monitor usage in the metrics section
        
        ### Support
        
        - [Fluence Documentation](https://fluence.dev/docs)
        - [Fluence Console](https://console.fluence.network)
        - [Community Discord](https://discord.gg/fluence)
        """)


def render_fluence_integration_main():
    """Main Fluence integration UI"""
    # Configuration modal
    render_fluence_config_modal()
    
    # Compute section
    render_fluence_compute_section()
    
    # Metrics
    render_fluence_metrics()
    
    # Help
    render_fluence_help()
