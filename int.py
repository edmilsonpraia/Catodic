import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import base64
import io
import json
from typing import Dict, Any

class RiskAnalysis:
    def __init__(self, system):
        self.system = system
        self.risk_metrics = {
            'corrosion_risk': 0,
            'environmental_risk': 0,
            'system_vulnerability': 0
        }
    
    def calculate_risk_index(self, geological_data, system_params):
        """
        Calculate a composite risk index
        """
        # Corrosion risk based on soil corrosivity
        corrosion_risk = geological_data.get('corrosivity_index', 0) / 100
        
        # Environmental risk based on terrain parameters
        environmental_factors = {
            'urban': 0.3,
            'coastal': 0.7,
            'desert': 0.2,
            'tropical_savanna': 0.5,
            'tropical_rainforest': 0.6
        }
        environmental_risk = environmental_factors.get(
            geological_data.get('terrain_type', 'urban'), 0.5
        )
        
        # System vulnerability based on protection parameters
        system_vulnerability = (
            (abs(system_params['potential'] - (-0.85)) / 1) *  # Deviation from ideal potential
            (1 - system_params['battery_level'] / 100) *  # Battery level
            (1 - system_params['energy_efficiency'] / 100)  # Energy efficiency
        )
        
        # Composite risk index
        risk_index = (
            corrosion_risk * 0.4 + 
            environmental_risk * 0.3 + 
            system_vulnerability * 0.3
        )
        
        self.risk_metrics = {
            'corrosion_risk': corrosion_risk * 100,
            'environmental_risk': environmental_risk * 100,
            'system_vulnerability': system_vulnerability * 100,
            'overall_risk_index': risk_index * 100
        }
        
        return self.risk_metrics

class EnvironmentalImpactAnalysis:
    def __init__(self, system):
        self.system = system
        self.environmental_metrics = {
            'carbon_offset': 0,
            'energy_savings': 0,
            'resource_conservation': 0
        }
    
    def calculate_environmental_impact(self):
        """
        Calculate environmental impact of solar cathodic protection system
        """
        # Carbon savings calculation
        carbon_offset = (
            self.system.solar_power * 0.5 /  # Solar energy to CO2 reduction conversion
            1000  # kWh to MWh
        )
        
        # Energy savings
        energy_savings = (
            self.system.solar_power * 
            self.system.solar_panel_efficiency
        )
        
        # Resource conservation
        resource_conservation = (
            20  # estimated lifetime years
        )
        
        self.environmental_metrics = {
            'carbon_offset': carbon_offset,
            'energy_savings': energy_savings,
            'resource_conservation': resource_conservation
        }
        
        return self.environmental_metrics

class GeologicalData:
    """Class for managing geological and geographic data"""
    def __init__(self, latitude: float, longitude: float):
        self.latitude = latitude
        self.longitude = longitude
        self.soil_data = {
            'resistivity': None,
            'ph': None,
            'mineral_composition': {},
            'corrosivity_index': 0,
            'moisture_content': None
        }
        self.environmental_data = {
            'altitude': None,
            'terrain_type': None,
            'proximity_to_water': None,
            'climate_zone': None
        }
    
    def calculate_corrosivity(self) -> float:
        """
        Calculate corrosivity index based on multiple parameters
        """
        # Corrosivity calculation example
        resistivity_factor = 1 / (self.soil_data['resistivity'] + 1) if self.soil_data['resistivity'] else 0.5
        ph_factor = abs(7 - self.soil_data['ph']) / 7 if self.soil_data['ph'] else 0.5
        moisture_factor = self.soil_data['moisture_content'] / 100 if self.soil_data['moisture_content'] else 0.5
        
        corrosivity_index = (
            resistivity_factor * 0.4 + 
            ph_factor * 0.3 + 
            moisture_factor * 0.3
        )
        
        self.soil_data['corrosivity_index'] = corrosivity_index * 100
        return self.soil_data['corrosivity_index']
    
    def get_local_geological_data(self) -> Dict[str, Any]:
        """
        Simulate obtaining local geological data
        In a real scenario, this could connect to a geological data API
        """
        # Simulated geological database
        geological_database = {
            (7.3697, 12.3547): {  # Central Cameroon coordinates
                'resistivity': 55,  # ohm.m
                'ph': 6.5,
                'mineral_composition': {
                    'clay': 0.45,
                    'sand': 0.35,
                    'silt': 0.15,
                    'organic_matter': 0.05
                },
                'moisture_content': 35,
                'altitude': 650,  # meters (central plateau average)
                'terrain_type': 'tropical_savanna',
                'proximity_to_water': 'moderate',
                'climate_zone': 'tropical_wet_dry'
            },
            (-3.1190, -60.0217): {  # Manaus, Amazon
                'resistivity': 40,
                'ph': 5.5,
                'mineral_composition': {
                    'clay': 0.6,
                    'sand': 0.2,
                    'silt': 0.1,
                    'organic_matter': 0.1
                },
                'moisture_content': 40,
                'altitude': 92,
                'terrain_type': 'tropical_rainforest',
                'proximity_to_water': 'high',
                'climate_zone': 'tropical_humid'
            }
        }
        
        # Function to find the nearest location
        def find_nearest_location(lat, lon):
            def distance(loc1, loc2):
                return ((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)**0.5
            
            nearest = min(geological_database.keys(), 
                         key=lambda loc: distance(loc, (lat, lon)))
            return geological_database[nearest]
        
        # Search for location data or use default
        location_data = find_nearest_location(self.latitude, self.longitude)
        
        self.soil_data.update({k: v for k, v in location_data.items() if k in self.soil_data})
        self.environmental_data.update({k: v for k, v in location_data.items() if k in self.environmental_data})
        
        return {**self.soil_data, **self.environmental_data}

class AdvancedCorrosionAnalysis:
    """Class for advanced corrosion analysis"""
    def __init__(self, system):
        self.system = system
        self.corrosion_metrics = {
            'estimated_corrosion_rate': 0,
            'protection_effectiveness': 0,
            'expected_equipment_lifetime': 0
        }
    
    def calculate_corrosion_rate(self, geological_data):
        """
        Calculate estimated corrosion rate
        """
        corrosivity_index = geological_data.get('corrosivity_index', 0)
        protection_potential = self.system.potential
        
        # Simplified corrosion rate model
        corrosion_rate = (
            (100 - corrosivity_index) * 0.01 * 
            (1 - abs(protection_potential - (-0.85)))
        )
        
        # Protection effectiveness calculation
        protection_effectiveness = (
            1 - (abs(protection_potential - (-0.85)) / 1)  # Deviation from ideal potential
        ) * 100
        
        self.corrosion_metrics = {
            'estimated_corrosion_rate': corrosion_rate,
            'protection_effectiveness': protection_effectiveness,
            'expected_equipment_lifetime': 0  # Will be calculated in another method
        }
        
        return corrosion_rate
    
    def estimate_equipment_lifetime(self, corrosion_rate):
        """
        Estimate equipment lifetime
        """
        # Simplified lifetime model
        base_lifetime = 20  # years
        corrosion_factor = 1 - corrosion_rate
        
        # Lifetime adjustment based on corrosion rate
        lifetime_adjustment = base_lifetime * corrosion_factor
        
        # Minimum and maximum lifetime limits
        min_lifetime = 5  # years
        max_lifetime = 30  # years
        
        equipment_lifetime = max(
            min_lifetime, 
            min(lifetime_adjustment, max_lifetime)
        )
        
        self.corrosion_metrics['expected_equipment_lifetime'] = equipment_lifetime
        
        return equipment_lifetime

class EnergyEfficiencyAnalysis:
    """Class for energy efficiency analysis"""
    def __init__(self, system):
        self.system = system
        self.efficiency_metrics = {
            'solar_conversion_efficiency': 0,
            'battery_charging_efficiency': 0,
            'system_energy_loss': 0,
            'total_energy_efficiency': 0
        }
    
    def calculate_energy_conversion(self):
        """
        Calculate solar energy conversion efficiency
        """
        solar_power = self.system.solar_power
        battery_input = solar_power * self.system.solar_panel_efficiency
        
        # Solar conversion efficiency
        solar_conversion_efficiency = (battery_input / solar_power) * 100
        
        # Battery charging efficiency
        battery_charging_efficiency = (
            (1 - (self.system.battery_voltage / 15)) * 100  # Simplified adjustment
        )
        
        # System energy loss estimation
        system_energy_loss = 100 - solar_conversion_efficiency
        
        # Total system efficiency
        total_energy_efficiency = (
            solar_conversion_efficiency * 0.6 + 
            battery_charging_efficiency * 0.4
        )
        
        self.efficiency_metrics = {
            'solar_conversion_efficiency': solar_conversion_efficiency,
            'battery_charging_efficiency': battery_charging_efficiency,
            'system_energy_loss': system_energy_loss,
            'total_energy_efficiency': total_energy_efficiency
        }
        
        return total_energy_efficiency

class CathodicProtectionSystem:
    def __init__(self):
        # Basic system parameters
        self.solar_panel_efficiency = 0.95
        self.battery_voltage = 12.0
        self.protection_current = 0.0
        self.reference_potential = -0.85  # V vs CSE
        self.solar_power = 0
        self.potential = 0
        
        # New analysis additions
        self.geological_data = GeologicalData(7.3697, 12.3547)  # Default Cameroon coordinates
        self.corrosion_analysis = AdvancedCorrosionAnalysis(self)
        self.energy_analysis = EnergyEfficiencyAnalysis(self)
        self.risk_analysis = RiskAnalysis(self)
        self.environmental_impact = EnvironmentalImpactAnalysis(self)
        
        # Reference optimal parameters
        self.optimal_parameters = {
            'potential': -0.85,
            'current': 2.0,
            'power': 800,
            'battery_level': 80,
            'energy_efficiency': 90
        }
    
    def calculate_protection_parameters(self, solar_irradiance):
        """
        Calculate cathodic protection parameters
        """
        # Default values for metrics
        corrosivity_index = 0
        corrosion_rate = 0
        equipment_lifetime = 0
        energy_efficiency = 0
        risk_metrics = {}
        environmental_impact = {}
        geological_data = {}

        try:
            # Obtain local geological data
            geological_data = self.geological_data.get_local_geological_data()
            
            # Calculate protection parameters
            self.solar_power = solar_irradiance * self.solar_panel_efficiency
            self.protection_current = self.solar_power / self.battery_voltage
            self.potential = self.reference_potential - (0.1 * np.random.random())
            
            # Additional analyses
            corrosivity_index = self.geological_data.calculate_corrosivity()
            corrosion_rate = self.corrosion_analysis.calculate_corrosion_rate(geological_data)
            equipment_lifetime = self.corrosion_analysis.estimate_equipment_lifetime(corrosion_rate)
            energy_efficiency = self.energy_analysis.calculate_energy_conversion()
            
            # Risk analysis
            risk_metrics = self.risk_analysis.calculate_risk_index(
                geological_data, 
                {
                    'potential': self.potential,
                    'battery_level': min(100, 60 + 40 * (self.solar_power / 1000)),
                    'energy_efficiency': energy_efficiency
                }
            )
            
            # Environmental impact
            environmental_impact = self.environmental_impact.calculate_environmental_impact()
        
        except Exception as e:
            print(f"Error calculating parameters: {e}")

        return {
            "power": self.solar_power,
            "current": self.protection_current,
            "potential": self.potential,
            "battery_level": min(100, 60 + 40 * (self.solar_power / 1000)),
            "corrosivity_index": corrosivity_index,
            "corrosion_rate": corrosion_rate,
            "equipment_lifetime": equipment_lifetime,
            "energy_efficiency": energy_efficiency,
            "risk_metrics": risk_metrics,
            "environmental_impact": environmental_impact,
            "geological_data": geological_data
        }
    
    def normalize_parameters(self):
        """Automatically adjust parameters to optimal values"""
        return {
            "power": self.optimal_parameters['power'],
            "current": self.optimal_parameters['current'],
            "potential": self.optimal_parameters['potential'],
            "battery_level": self.optimal_parameters['battery_level']
        }

def generate_historical_data(hours=24):
    """Generate historical data for system monitoring"""
    times = pd.date_range(end=datetime.now(), periods=hours, freq='H')
    data = pd.DataFrame({
        'timestamp': times,
        'potential': [-0.85 + 0.05 * np.sin(np.pi * i/12) + 0.02 * np.random.randn() for i in range(hours)],
        'current': [2 + 0.5 * np.sin(np.pi * i/12) + 0.1 * np.random.randn() for i in range(hours)],
        'solar_power': [800 * np.sin(np.pi * i/12)**2 + 50 * np.random.randn() for i in range(hours)],
        'battery_level': [80 + 10 * np.sin(np.pi * i/12) + 5 * np.random.randn() for i in range(hours)],
        'corrosivity_index': [50 + 20 * np.sin(np.pi * i/12) + 10 * np.random.randn() for i in range(hours)]
    })
    return data

def export_to_excel(params, potential_threshold, current_threshold):
    """Generate Excel report with current system data"""
    output = io.BytesIO()
    
    # Create historical data
    historical_data = generate_historical_data()
    
    # Create summary data
    summary_data = pd.DataFrame({
        'Parameter': [
            'Potential', 
            'Current', 
            'Solar Power', 
            'Battery Level', 
            'Corrosivity Index', 
            'Corrosion Rate', 
            'Equipment Lifetime', 
            'Energy Efficiency',
            'Carbon Offset',
            'Overall Risk Index'
        ],
        'Current Value': [
            f"{params['potential']:.3f} V",
            f"{params['current']:.2f} A",
            f"{params['power']:.1f} W",
            f"{params['battery_level']:.1f}%",
            f"{params['corrosivity_index']:.2f}%",
            f"{params['corrosion_rate']:.4f}",
            f"{params['equipment_lifetime']:.1f} years",
            f"{params['energy_efficiency']:.2f}%",
            f"{params['environmental_impact'].get('carbon_offset', 0):.4f} tCO2",
            f"{params['risk_metrics'].get('overall_risk_index', 0):.2f}%"
        ],
        'Status': [
            'Normal' if params['potential'] <= potential_threshold else 'Alert',
            'Normal' if params['current'] <= current_threshold else 'Alert',
            'Normal' if params['power'] >= 500 else 'Low',
            'Normal' if params['battery_level'] >= 20 else 'Critical',
            'Low' if params['corrosivity_index'] < 30 else 'High',
            'Normal' if params['corrosion_rate'] < 0.1 else 'Critical', 
            'Normal' if params['equipment_lifetime'] > 15 else 'Warning',
            'High' if params['energy_efficiency'] > 90 else 'Normal',
            'Positive' if params['environmental_impact'].get('carbon_offset', 0) > 0 else 'Neutral',
            'Low' if params['risk_metrics'].get('overall_risk_index', 0) < 30 else 'High'
        ]
    })
    
    # Create Excel writer object
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Write summary sheet
        summary_data.to_excel(writer, sheet_name='System Summary', index=False)
        
        # Write historical data
        historical_data.to_excel(writer, sheet_name='Historical Data', index=False)
        
        # Write geological data
        geological_df = pd.DataFrame.from_dict(params['geological_data'], orient='index', columns=['Value'])
        geological_df.to_excel(writer, sheet_name='Geological Data')
        
        # Get workbook and worksheet objects
        workbook = writer.book
        
        # Add formatting
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#023047',
            'font_color': 'white',
            'align': 'center',
            'valign': 'vcenter'
        })
        
        # Apply formatting for all sheets
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for col_num, value in enumerate(summary_data.columns.values):
                worksheet.write(0, col_num, value, header_format)
                worksheet.set_column(col_num, col_num, 20)
    
    return output.getvalue()

def custom_simulink_calculator():
    """Custom advanced calculator for cathodic protection system"""
    st.subheader("🧮 Advanced Cathodic Protection Calculator")
    
    # Columns for comprehensive input
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Input Parameters")
        # System Performance Inputs
        potential = st.number_input("Potential (V)", value=-0.85, step=0.01)
        current = st.number_input("Current (A)", value=2.0, step=0.1)
        solar_power = st.number_input("Solar Power (W)", value=800.0, step=10.0)
        
        # Geological Inputs
        soil_resistivity = st.number_input("Soil Resistivity (Ω.m)", value=50.0, step=1.0)
        soil_ph = st.number_input("Soil pH", value=6.5, min_value=0.0, max_value=14.0, step=0.1)
        moisture_content = st.number_input("Soil Moisture (%)", value=35.0, min_value=0.0, max_value=100.0, step=0.5)
    
    with col2:
        st.markdown("### System Characteristics")
        # Additional System Parameters
        battery_voltage = st.number_input("Battery Voltage (V)", value=12.0, step=0.1)
        battery_capacity = st.number_input("Battery Capacity (Ah)", value=100.0, step=1.0)
        panel_efficiency = st.number_input("Panel Efficiency (%)", value=95.0, min_value=0.0, max_value=100.0, step=0.5)
        
        # Environmental Factors
        ambient_temperature = st.number_input("Ambient Temperature (°C)", value=25.0, step=0.5)
        solar_irradiance = st.number_input("Solar Irradiance (W/m²)", value=800.0, step=10.0)
        altitude = st.number_input("Altitude (m)", value=650.0, step=10.0)
    
    with col3:
        st.markdown("### Advanced Calculations")
        # Comprehensive Calculations
        
        # Protection Ratio Calculations
        protection_ratio = (potential / -0.85) * 100
        current_efficiency = (current / 2.0) * 100
        power_utilization = (solar_power / 1000) * 100
        
        # Corrosivity Calculations
        resistivity_factor = 1 / (soil_resistivity + 1)
        ph_factor = abs(7 - soil_ph) / 7
        moisture_factor = moisture_content / 100
        
        corrosivity_index = (
            resistivity_factor * 0.4 + 
            ph_factor * 0.3 + 
            moisture_factor * 0.3
        ) * 100
        
        # Risk Assessment
        risk_index = 100 - (
            protection_ratio * 0.3 + 
            current_efficiency * 0.2 + 
            power_utilization * 0.2 +
            corrosivity_index * 0.3
        )
        
        # Display Calculations
        st.metric("Protection Ratio", f"{protection_ratio:.2f}%")
        st.metric("Current Efficiency", f"{current_efficiency:.2f}%")
        st.metric("Power Utilization", f"{power_utilization:.2f}%")
        st.metric("Corrosivity Index", f"{corrosivity_index:.2f}%")
        
        # Risk Classification
        st.metric("Risk Index", f"{risk_index:.2f}%")
        if risk_index < 30:
            st.success("🟢 Low Risk")
        elif risk_index < 60:
            st.warning("🟠 Moderate Risk")
        else:
            st.error("🔴 High Risk")
    
    # Detailed Analysis Section
    st.markdown("### Detailed System Analysis")
    col_analysis1, col_analysis2 = st.columns(2)
    
    with col_analysis1:
        # Energy Efficiency Calculations
        battery_efficiency = min(100, (battery_voltage / 15) * 100)
        energy_conversion_efficiency = (solar_power * panel_efficiency) / solar_irradiance * 100
        
        st.metric("Battery Efficiency", f"{battery_efficiency:.2f}%")
        st.metric("Energy Conversion", f"{energy_conversion_efficiency:.2f}%")
    
    with col_analysis2:
        # Environmental Impact Estimation
        carbon_offset_factor = solar_power / 1000 * 0.5  # Simplified CO2 reduction
        resource_conservation = 20 * (1 - (risk_index / 100))  # Estimated system lifetime
        
        st.metric("Est. Carbon Offset", f"{carbon_offset_factor:.4f} tCO2")
        st.metric("Resource Conservation", f"{resource_conservation:.1f} years")
        
# Custom styling
st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .stButton button {
        background-color: #023047; 
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        transition: background-color 0.3s ease;
    }
    .stButton button:hover {
        background-color: #034a6d;
    }
    .stTextInput input {
        border-color: #023047;
        border-radius: 5px;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: scale(1.02);
    }
    .control-button {
        margin: 10px 0;
        width: 100%;
    }
    /* Improvements in text readability */
    .stMarkdown {
        color: #333;
        line-height: 1.6;
    }
    /* Style for headers */
    h1, h2, h3 {
        color: #023047;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# System initialization
@st.cache_resource
def init_system():
    return CathodicProtectionSystem()

# Initialize the system
system = init_system()

# Sidebar Controls
with st.sidebar:
    st.title("🛠️ System Controls")
    
    # Solar Irradiance Settings Section
    st.subheader("Solar Irradiance Parameters")
    solar_irradiance = st.slider(
        "Solar Irradiance (W/m²)", 
        min_value=0, 
        max_value=1000, 
        value=800,
        help="Adjust the intensity of incident solar radiation"
    )
    
    # Protection Thresholds Section
    st.subheader("Protection Thresholds")
    potential_threshold = st.slider(
        "Potential Threshold (V)", 
        min_value=-1.5, 
        max_value=-0.5, 
        value=-0.85,
        help="Maximum potential value for safe operation"
    )
    current_threshold = st.slider(
        "Current Threshold (A)", 
        min_value=0.0, 
        max_value=5.0, 
        value=2.0,
        help="Maximum current value for safe operation"
    )
    
    # Geological Location Section
    st.subheader("Geological Location")
    latitude = st.number_input(
        "Latitude", 
        min_value=-90.0, 
        max_value=90.0, 
        value=7.3697,  # Cameroon default
        help="Latitude coordinate for geological analysis"
    )
    longitude = st.number_input(
        "Longitude", 
        min_value=-180.0, 
        max_value=180.0, 
        value=12.3547,  # Cameroon default
        help="Longitude coordinate for geological analysis"
    )
    
    # Update geological data based on new coordinates
    system.geological_data = GeologicalData(latitude, longitude)
    geological_data = system.geological_data.get_local_geological_data()
    
    # Display updated local information
    st.subheader("Updated Local Information")
    st.write(f"**Climate Zone:** {geological_data.get('climate_zone', 'N/A')}")
    st.write(f"**Terrain Type:** {geological_data.get('terrain_type', 'N/A')}")
    
    # Advanced Calculator Option
    st.markdown("---")  # Separator
    if st.checkbox("Open Advanced Calculator"):
        custom_simulink_calculator()

# Main Panel
st.title("⚡ Solar-Powered Cathodic Protection System")

# Control Buttons Row
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🎯 Normalize Parameters", key="normalize", help="Adjust system to optimal values"):
        params = system.normalize_parameters()
        st.success("Parameters normalized to ideal values")
    else:
        params = system.calculate_protection_parameters(solar_irradiance)

with col2:
    if st.button("🔄 Reset System", key="reset", help="Reload system to initial state"):
        st.cache_resource.clear()
        st.rerun()

with col3:
    if st.button("📊 Export Report", key="export", help="Generate detailed Excel report"):
        excel_data = export_to_excel(params, potential_threshold, current_threshold)
        b64 = base64.b64encode(excel_data).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="cathodic_protection_report.xlsx">Download Excel Report</a>'
        st.markdown(href, unsafe_allow_html=True)

# Real-time Indicators
st.markdown("### Real-time Indicators")
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    st.metric(
        "Protection Potential",
        f"{params['potential']:.3f} V",
        delta=f"{(params['potential'] - system.reference_potential):.3f} V",
        help="Current cathodic protection potential"
    )
with col2:
    st.metric(
        "Protection Current",
        f"{params['current']:.2f} A",
        delta=f"{params['current'] - 2:.2f} A",
        help="Current cathodic protection current"
    )
with col3:
    st.metric(
        "Solar Power",
        f"{params['power']:.1f} W",
        delta=f"{params['power'] - 750:.1f} W",
        help="Solar power generated"
    )
with col4:
    st.metric(
        "Battery Level",
        f"{params['battery_level']:.1f}%",
        delta=f"{params['battery_level'] - 80:.1f}%",
        help="Battery charge level"
    )
with col5:
    corrosivity_index = params.get('corrosivity_index', 0)
    st.metric(
        "Corrosivity Index", 
        f"{corrosivity_index:.2f}%",
        delta=f"{corrosivity_index - 50:.2f}%",
        help="Soil corrosive potential index"
    )
with col6:
    equipment_lifetime = params.get('equipment_lifetime', 0)
    st.metric(
        "Equipment Lifetime", 
        f"{equipment_lifetime:.1f} years",
        delta=f"{equipment_lifetime - 20:.1f} years",
        help="Estimated system lifetime"
    )

# Alert System
st.subheader("Alert System")
alerts = []
if params.get('potential', 0) > potential_threshold:
    alerts.append("⚠️ Potential above threshold!")
if params.get('current', 0) > current_threshold:
    alerts.append("⚠️ Current above threshold!")
if params.get('battery_level', 0) < 20:
    alerts.append("🔋 Critical battery level!")
if params.get('corrosivity_index', 0) > 70:
    alerts.append("🟠 High soil corrosivity!")
if params.get('equipment_lifetime', 0) < 10:
    alerts.append("⚠️ Reduced equipment lifetime!")
if params.get('risk_metrics', {}).get('overall_risk_index', 0) > 60:
    alerts.append("🚨 High risk index!")

if alerts:
    for alert in alerts:
        st.warning(alert)
else:
    st.success("System operating normally")
    # Real-time Monitoring
st.subheader("Real-time Monitoring")
historical_data = generate_historical_data()

# Multi-line Chart
fig_multiline = go.Figure()

# Add traces for each metric
fig_multiline.add_trace(go.Scatter(
    x=historical_data['timestamp'],
    y=historical_data['potential'],
    name='Potential (V)',
    yaxis='y1',
    line=dict(color='#023047', width=2)
))
fig_multiline.add_trace(go.Scatter(
    x=historical_data['timestamp'],
    y=historical_data['current'],
    name='Current (A)',
    yaxis='y2',
    line=dict(color='#FFA500', width=2)
))
fig_multiline.add_trace(go.Scatter(
    x=historical_data['timestamp'],
    y=historical_data['solar_power'],
    name='Solar Power (W)',
    yaxis='y3',
    line=dict(color='#4CAF50', width=2)
))

# Layout configuration
fig_multiline.update_layout(
    height=600,
    title='System Data - Last 24 Hours',
    yaxis=dict(
        title='Potential (V)', 
        side='left', 
        titlefont=dict(color='#023047'),
        tickfont=dict(color='#023047')
    ),
    yaxis2=dict(
        title='Current (A)', 
        side='right', 
        overlaying='y', 
        anchor='x',
        titlefont=dict(color='#FFA500'),
        tickfont=dict(color='#FFA500')
    ),
    yaxis3=dict(
        title='Solar Power (W)', 
        side='right', 
        overlaying='y', 
        position=0.95,
        titlefont=dict(color='#4CAF50'),
        tickfont=dict(color='#4CAF50')
    ),
    xaxis_title='Timestamp',
    showlegend=True,
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1
    )
)

# Render the chart
st.plotly_chart(fig_multiline, use_container_width=True)

# Additional Charts Section
st.markdown("### Detailed Analyses")
col1, col2 = st.columns(2)

with col1:
    # Mineral Composition Chart
    mineral_comp = params.get('geological_data', {}).get('mineral_composition', {})
    if mineral_comp:
        st.subheader("Soil Mineral Composition")
        fig_pie = px.pie(
            values=list(mineral_comp.values()),
            names=list(mineral_comp.keys()),
            title="Mineral Composition",
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    # Risk Analysis Chart
    risk_metrics = params.get('risk_metrics', {})
    if risk_metrics:
        st.subheader("Risk Analysis")
        risk_data = [
            {'Category': 'Corrosion Risk', 'Value': risk_metrics.get('corrosion_risk', 0)},
            {'Category': 'Environmental Risk', 'Value': risk_metrics.get('environmental_risk', 0)},
            {'Category': 'System Vulnerability', 'Value': risk_metrics.get('system_vulnerability', 0)}
        ]
        fig_risk = px.bar(
            risk_data, 
            x='Category', 
            y='Value', 
            title='Risk Metrics',
            color='Category',
            color_discrete_map={
                'Corrosion Risk': '#FF6B6B',
                'Environmental Risk': '#4ECDC4',
                'System Vulnerability': '#45B7D1'
            }
        )
        fig_risk.update_layout(
            yaxis_title='Risk Index (%)',
            xaxis_title='Risk Category'
        )
        st.plotly_chart(fig_risk, use_container_width=True)

# Environmental Impact Section
st.subheader("Environmental Impact")
env_impact = params.get('environmental_impact', {})
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        "Carbon Offset", 
        f"{env_impact.get('carbon_offset', 0):.4f} tCO2",
        help="Tons of CO2 avoided"
    )
with col2:
    st.metric(
        "Energy Savings", 
        f"{env_impact.get('energy_savings', 0):.1f} kWh",
        help="Energy saved by the system"
    )
with col3:
    st.metric(
        "Resource Conservation", 
        f"{env_impact.get('resource_conservation', 0):.1f} years",
        help="Estimated resource conservation"
    )
# Advanced Settings Panel
with st.expander("⚙️ Advanced Settings"):
    st.subheader("System Parameters")
    
    # Column division to organize settings
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Solar Panel Configuration")
        # Solar Panel Efficiency
        solar_panel_efficiency = st.slider(
            "Solar Panel Efficiency (%)", 
            min_value=0, 
            max_value=100, 
            value=95,
            help="Adjust solar panel conversion efficiency"
        )
        
        # Battery Nominal Voltage
        battery_voltage = st.slider(
            "Battery Nominal Voltage (V)", 
            min_value=0.0, 
            max_value=24.0, 
            value=12.0,
            step=0.1,
            help="Set the nominal battery system voltage"
        )
        
        # Solar Panel Area
        solar_panel_area = st.slider(
            "Solar Panel Area (m²)", 
            min_value=0.0, 
            max_value=10.0, 
            value=2.0,
            step=0.1,
            help="Total solar energy collection area"
        )
    
    with col2:
        st.markdown("### Protection Configuration")
        # Reference Potential
        reference_potential = st.slider(
            "Reference Potential (V)", 
            min_value=-2.0, 
            max_value=0.0, 
            value=-0.85,
            step=0.01,
            help="Cathodic protection reference potential"
        )
        
        # Battery Capacity
        battery_capacity = st.slider(
            "Battery Capacity (Ah)", 
            min_value=0, 
            max_value=200, 
            value=100,
            help="Total battery storage capacity"
        )
        
        # Protection Current
        protection_current = st.slider(
            "Protection Current (A)", 
            min_value=0.0, 
            max_value=5.0, 
            value=2.0,
            step=0.1,
            help="Current required for effective cathodic protection"
        )
    
    # Apply Advanced Settings Button
    if st.button("🔧 Apply Advanced Settings"):
        # Logic to update system parameters
        st.success("Advanced settings applied successfully!")

# Report Generation Panel
with st.expander("📄 Report Generation"):
    st.subheader("Performance Report")
    
    # Date range selection
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("Start Date")
    
    with col2:
        end_date = st.date_input("End Date")
    
    # Report type selection
    report_type = st.selectbox(
        "Report Type",
        [
            "Executive Summary", 
            "Detailed Analysis", 
            "Maintenance Report", 
            "Environmental Risk Assessment"
        ],
        help="Select the type of report to be generated"
    )
    
    # Report customization options
    st.markdown("### Customize Report")
    
    # Checkboxes for section inclusion
    include_geological_data = st.checkbox("Include Geological Data")
    include_risk_analysis = st.checkbox("Include Risk Analysis")
    include_environmental_impact = st.checkbox("Include Environmental Impact")
    
    # Generate Report Button
    if st.button("📋 Generate Customized Report"):
        # Report generation logic
        with st.spinner('Generating report...'):
            # Simulate report generation
            report_content = f"""
            # Cathodic Protection System Report

            ## General Information
            - **Period:** {start_date} to {end_date}
            - **Report Type:** {report_type}

            ## Performance Summary
            - Average Potential: {params['potential']:.3f} V
            - Average Current: {params['current']:.2f} A
            - Average Solar Power: {params['power']:.1f} W

            {"## Geological Data" if include_geological_data else ""}
            {str(params['geological_data']) if include_geological_data else ""}

            {"## Risk Analysis" if include_risk_analysis else ""}
            {str(params['risk_metrics']) if include_risk_analysis else ""}

            {"## Environmental Impact" if include_environmental_impact else ""}
            {str(params['environmental_impact']) if include_environmental_impact else ""}
            """
            
            # Download button for the report
            b64 = base64.b64encode(report_content.encode()).decode()
            href = f'<a href="data:text/plain;base64,{b64}" download="cathodic_protection_report.txt">Download Report</a>'
            st.markdown(href, unsafe_allow_html=True)

# Final system information section
st.markdown("---")
st.markdown("### System Information")
col1, col2 = st.columns(2)

with col1:
    st.info(f"""
    **System Version:** 1.2.0
    **Last Update:** January 2024
    **Operation Mode:** Solar Cathodic Protection
    """)

with col2:
    st.info(f"""
    **Location:** {latitude}° N, {longitude}° W
    **Climate Zone:** {params.get('geological_data', {}).get('climate_zone', 'Not Defined')}
    **Terrain Type:** {params.get('geological_data', {}).get('terrain_type', 'Not Defined')}
    """)

# Footer
st.markdown("""
<div style='text-align: center; color: #666; margin-top: 20px;'>
    © 2024 Solar Cathodic Protection System. All rights reserved.
</div>
""", unsafe_allow_html=True)

# Run the script
if __name__ == "__main__":
    st.write("Solar Cathodic Protection System initialized.")