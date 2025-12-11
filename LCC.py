"""
Life Cycle Cost Model 
Last update: 12/11/2025

Authors: Dylan Hogge
Citation: Trani, A., Baik, H., Hinze, N., Ashiabor, S., Viken, J., and Dollyhigh, S., "Nationwide Impacts of Very Light Jet Traffic in the
Future Next Generation Air Transportation System (NGATS)," 6th AIAA Aviation Technology, Integration and Operations
Conference, Wichita, Kansas, 2006. https://doi.org/10.2514/6.2006-7763, session: ATIO-25: Airspace System Demand/Delay
Modeling II.

Inputs: Excel files with aircraft parameters and lookup tables
Outputs: Economic metrics and visualization
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from pathlib import Path


# CONSTANTS 
MILE_TO_NM = 1.151


@dataclass
class AnnualCosts:
    """Yearly cost breakdown for an aircraft configuration."""
    
    # Derived flight parameters
    pilots_per_aircraft: float
    aircraft_speed: float
    flight_time_hours: float
    flights_per_hr: float
    
    # Purchase/financing
    automation_cost: float
    aircraft_purchase_price: float
    resale_value: float
    
    # Fixed costs
    annual_amortization: float
    annual_depreciation: float
    landing_site_annual_support: float
    hull_insurance: float
    liability_insurance: float
    maintenance_software: float
    miscellaneous_service: float
    property_tax: float
    hangar_and_office_expenses: float
    personnel_benefits: float
    pilots_salaries: float
    staff_salaries: float
    annual_personnel_costs: float
    annual_training_cost: float
    annual_fixed_costs: float
    
    # Variable costs (per hour)
    maintenance_hours_per_flight_hour: float
    fuel_burn_per_hour: float
    fuel_cost_per_hour: float
    maintenance_labor_per_hour: float
    schedule_parts_per_hour: float
    midlife_inspection_per_hour: float
    propeller_allowance_per_hour: float
    engine_overhaul_per_hour: float
    modernisation_per_hour: float
    paint_per_hour: float
    refurbishing_per_hour: float
    battery_per_hour: float
    miscellaneous_trip_expenses_per_hour: float
    landing_fees_per_hour: float
    total_variable_cost_per_hour: float
    annual_variable_costs: float
    
    # Totals
    total_annual_costs: float


@dataclass
class LifeCycleResults:
    """Results from simulating costs over the aircraft's operational life"""
    
    annual_costs: AnnualCosts
    time_series: np.ndarray = field(repr=False)
    cumulative_costs: np.ndarray = field(repr=False)
    life_cycle_total_cost: float
    revenue_casm: float
    fare_per_pax_nm: float
    pax_seats: float


def load_parameters(param_file: str) -> Dict[str, Any]:
    """Load params and tables from excel."""
    param_table = pd.read_excel(param_file, sheet_name='Parameters')
    
    params = {}
    for _, row in param_table.iterrows():
        param_name = row.iloc[0]
        param_value = row.iloc[1]
        if pd.notna(param_name) and isinstance(param_name, str) and pd.notna(param_value):
            # Skips section headers
            try:
                if isinstance(param_value, (int, float)):
                    params[param_name] = float(param_value)
            except:
                pass
    
    # Lookup tables
    # These are not really detailed at the moment but future work intends to expand them    
    crews_lookup = pd.read_excel(param_file, sheet_name='Lookup_Crews', skiprows=2)
    speed_lookup = pd.read_excel(param_file, sheet_name='Lookup_Speed', skiprows=2)
    
    # Clean up the lookup tables
    crews_lookup = crews_lookup.dropna()
    speed_lookup = speed_lookup.dropna()
    
    params['crews_lookup_x'] = crews_lookup.iloc[:, 0].astype(float).values
    params['crews_lookup_y'] = crews_lookup.iloc[:, 1].astype(float).values
    params['speed_lookup_x'] = speed_lookup.iloc[:, 0].astype(float).values
    params['speed_lookup_y'] = speed_lookup.iloc[:, 1].astype(float).values
    
    return params


def calculate_annual_costs(params: Dict[str, Any]) -> AnnualCosts:
    """
    Calculate all annual cost components for an aircraft configuration.
    
    Returns a dataclass with fixed costs, variable costs, and derived parameters.
    """
    # Interpolations
    crews_interp = interp1d(params['crews_lookup_x'], params['crews_lookup_y'], 
                           kind='linear', fill_value='extrapolate')
    speed_interp = interp1d(params['speed_lookup_x'], params['speed_lookup_y'], 
                           kind='linear', fill_value='extrapolate')
    
    crews = float(crews_interp(params['Flight_Hours_per_Year']))
    pilots_per_aircraft = params['Number_of_Pilots'] * crews
    aircraft_speed = float(speed_interp(params['Mission_Stage_Length']))
    flight_time_hours = params['Mission_Stage_Length'] / aircraft_speed
    flights_per_hr = 1 / flight_time_hours
    
    # Automation cost
    if params['Number_of_Pilots'] == 0:
        automation_cost = params['Automation_Cost_Base']
    else:
        automation_cost = 0
    
    aircraft_purchase_price = params['Aircraft_Baseline_Cost'] + automation_cost
    resale_value = aircraft_purchase_price * params['Percent_Resale_Value']
    
    # Loan calculations
    loan_amount = aircraft_purchase_price * params['Finance_Percent']
    payments = params['Life_Cycle_Time'] * 12
    monthly_rate = params['Interest_Rate'] / 12
    
    if monthly_rate > 0 and loan_amount > 0:
        monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**payments) / \
                         ((1 + monthly_rate)**payments - 1)
    elif loan_amount > 0:
        monthly_payment = loan_amount / payments
    else:
        monthly_payment = 0
    
    # =========================================================================
    # FIXED COSTS
    # =========================================================================
    annual_amortization = monthly_payment * 12
    annual_depreciation = (aircraft_purchase_price - resale_value) / params['Life_Cycle_Time']
    landing_site_annual_support = params['Landing_Site_Support_Hourly_Cost'] * params['Flight_Hours_per_Year']
    hull_insurance = params['Aircraft_Baseline_Cost'] * params['Hull_Insurance_Rate']
    
    if params['Number_of_Pilots'] == 0:
        liability_insurance = (params['Liability_Insurance_Rate'] * params['Aircraft_Baseline_Cost']) * 1.1
    else:
        liability_insurance = params['Liability_Insurance_Rate'] * params['Aircraft_Baseline_Cost']
    
    maintenance_software = params['Maintenance_Software_Programs']
    miscellaneous_service = params['Miscellaneous_Service']
    property_tax = params['Property_Tax']
    hangar_and_office_expenses = params['Hangar_and_Office_Lease_Space'] + params['Miscellaneous_Office_Expense']
    
    personnel_benefits = params['Annual_Pilot_Salary'] * params['Percent_Salaries_to_Benefits'] * pilots_per_aircraft
    pilots_salaries = params['Annual_Pilot_Salary'] * pilots_per_aircraft
    staff_salaries = params['Loaded_Salary_of_Staff_Member'] * params['Staff_members_per_Vehicle']
    annual_personnel_costs = personnel_benefits + pilots_salaries + staff_salaries
    
    annual_training_cost = params['Recurrent_Maintenance_Training'] + \
                           params['Recurrent_Pilot_Training'] * pilots_per_aircraft
    
    annual_fixed_costs = (annual_amortization + landing_site_annual_support + 
                          hull_insurance + liability_insurance + 
                          maintenance_software + miscellaneous_service +
                          property_tax + hangar_and_office_expenses + 
                          annual_personnel_costs + annual_training_cost + 
                          annual_depreciation)
    
    # =========================================================================
    # VARIABLE COSTS
    # =========================================================================
    if automation_cost == 0:
        maintenance_hours_per_flight_hour = params['Baseline_MaintHours_per_FH']
    else:
        maintenance_hours_per_flight_hour = params['Baseline_MaintHours_per_FH'] * 1.0
    
    fuel_burn_per_hour = params['Gallons_per_Hour']
    fuel_cost_per_hour = fuel_burn_per_hour * params['Price_per_Gallon']
    maintenance_labor_per_hour = maintenance_hours_per_flight_hour * params['Maintenance_Labor_Expense_per_Hour']
    schedule_parts_per_hour = params['Schedule_Parts_Expense']
    midlife_inspection_per_hour = params['MidLife_Engine_Section_Inspection_Cost']
    propeller_allowance_per_hour = params['Propeller_Allowance']
    engine_overhaul_per_hour = (params['Engine_Overhaul_Cost'] * params['Number_of_Engines']) / params['Engine_Overhaul_Interval']
    modernisation_per_hour = params['Modernisation_and_Upgrades'] / params['Modernisation_Time_Interval']
    paint_per_hour = params['Aircraft_Paint'] / params['Paint_and_Refurbishing_Interval']
    refurbishing_per_hour = params['Interior_Refurbishing'] / params['Paint_and_Refurbishing_Interval']
    battery_per_hour = params['Battery_Cost'] / params['Battery_Replacement_Interval']
    miscellaneous_trip_expenses_per_hour = params['Miscellaneous_Trip_Expenses']
    landing_fees_per_hour = flights_per_hr * params['Landing_Fee_per_Landing']
    
    total_variable_cost_per_hour = (fuel_cost_per_hour + maintenance_labor_per_hour +
                                    schedule_parts_per_hour + midlife_inspection_per_hour +
                                    propeller_allowance_per_hour + engine_overhaul_per_hour +
                                    modernisation_per_hour + paint_per_hour +
                                    refurbishing_per_hour + battery_per_hour +
                                    miscellaneous_trip_expenses_per_hour + landing_fees_per_hour)
    
    annual_variable_costs = total_variable_cost_per_hour * params['Flight_Hours_per_Year']
    
    # Total annual costs
    total_annual_costs = annual_fixed_costs + annual_variable_costs
    
    return AnnualCosts(
        pilots_per_aircraft=pilots_per_aircraft,
        aircraft_speed=aircraft_speed,
        flight_time_hours=flight_time_hours,
        flights_per_hr=flights_per_hr,
        automation_cost=automation_cost,
        aircraft_purchase_price=aircraft_purchase_price,
        resale_value=resale_value,
        annual_amortization=annual_amortization,
        annual_depreciation=annual_depreciation,
        landing_site_annual_support=landing_site_annual_support,
        hull_insurance=hull_insurance,
        liability_insurance=liability_insurance,
        maintenance_software=maintenance_software,
        miscellaneous_service=miscellaneous_service,
        property_tax=property_tax,
        hangar_and_office_expenses=hangar_and_office_expenses,
        personnel_benefits=personnel_benefits,
        pilots_salaries=pilots_salaries,
        staff_salaries=staff_salaries,
        annual_personnel_costs=annual_personnel_costs,
        annual_training_cost=annual_training_cost,
        annual_fixed_costs=annual_fixed_costs,
        maintenance_hours_per_flight_hour=maintenance_hours_per_flight_hour,
        fuel_burn_per_hour=fuel_burn_per_hour,
        fuel_cost_per_hour=fuel_cost_per_hour,
        maintenance_labor_per_hour=maintenance_labor_per_hour,
        schedule_parts_per_hour=schedule_parts_per_hour,
        midlife_inspection_per_hour=midlife_inspection_per_hour,
        propeller_allowance_per_hour=propeller_allowance_per_hour,
        engine_overhaul_per_hour=engine_overhaul_per_hour,
        modernisation_per_hour=modernisation_per_hour,
        paint_per_hour=paint_per_hour,
        refurbishing_per_hour=refurbishing_per_hour,
        battery_per_hour=battery_per_hour,
        miscellaneous_trip_expenses_per_hour=miscellaneous_trip_expenses_per_hour,
        landing_fees_per_hour=landing_fees_per_hour,
        total_variable_cost_per_hour=total_variable_cost_per_hour,
        annual_variable_costs=annual_variable_costs,
        total_annual_costs=total_annual_costs
    )


def _dynamics(t: float, y: np.ndarray, params: Dict[str, Any], costs: AnnualCosts) -> np.ndarray:
    """
    ODE dynamics function for life cycle cost simulation.
    
    Args:
    
    Returns:

    """

    dydt = np.zeros(10)
    
    dydt[0] = costs.annual_amortization
    dydt[1] = costs.total_annual_costs 
    dydt[2] = costs.annual_fixed_costs
    dydt[3] = 0
    dydt[4] = 0
    dydt[5] = costs.annual_personnel_costs
    dydt[6] = costs.annual_training_cost
    dydt[7] = costs.annual_variable_costs
    dydt[8] = params['Flight_Hours_per_Year']  
    dydt[9] = costs.landing_fees_per_hour * params['Flight_Hours_per_Year']
    
    return dydt


def run_simulation(params: Dict[str, Any], costs: AnnualCosts) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the life cycle cost simulation.
    
    Args:
        params: Parameter dict from excel file
        costs: Pre-calculated annual costs
        
    Returns:
        Tuple of (time array, state array)
    """
    # Initial conditions
    if params['Number_of_Pilots'] == 0:
        automation_cost = params['Automation_Cost_Base']
    else:
        automation_cost = 0
    
    aircraft_purchase_price = params['Aircraft_Baseline_Cost'] + automation_cost
    
    # Down payment
    down_payment = aircraft_purchase_price * (1 - params['Finance_Percent'])
    
    # Initial training costs
    crews_interp = interp1d(params['crews_lookup_x'], params['crews_lookup_y'], 
                           kind='linear', fill_value='extrapolate')
    crews_init = float(crews_interp(params['Flight_Hours_per_Year']))
    pilots_per_aircraft_init = params['Number_of_Pilots'] * crews_init
    total_initial_training = (params['Initial_Pilot_Training'] * pilots_per_aircraft_init + 
                              params['Initial_Maintenance_Training'])
    
    # Initial state vector
    y0 = np.zeros(10)
    y0[1] = down_payment 
    y0[6] = total_initial_training 
    
    # Time span
    t_span = (0, params['Life_Cycle_Time'])
    t_eval = np.linspace(0, params['Life_Cycle_Time'], int(params['Life_Cycle_Time']) + 1)
    
    # Solve ODE
    solution = solve_ivp(
        fun=lambda t, y: _dynamics(t, y, params, costs),
        t_span=t_span,
        y0=y0,
        method='RK45',
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-8
    )
    
    return solution.t, solution.y


def calculate_life_cycle_results(params: Dict[str, Any]) -> LifeCycleResults:
    """    
    computes annual costs, runs the simulation, and calculates revenue metrics.
    
    Args:
        params: Parameter dictionary from excel file
        
    Returns:
        LifeCycleResults containing annual costs and simulation results
    """
    # Calculate annual costs
    costs = calculate_annual_costs(params)
    
    # Run simulation
    t, y = run_simulation(params, costs)
    
    # Life cycle total cost is the final cumulative cost
    life_cycle_total_cost = y[1, -1]
    
    # Revenue metrics
    deadhead_hours = params['Flight_Hours_per_Year'] * params['Percent_Repositioning_Flight_Hours'] / 100
    revenue_hours = params['Flight_Hours_per_Year'] - deadhead_hours
    revenue_nm_annual = revenue_hours * costs.aircraft_speed / MILE_TO_NM
    
    total_lc_revenue_nm = revenue_nm_annual * params['Life_Cycle_Time']
    
    # Revenue required per NM (with profit margin)
    revenue_required_per_nm = (life_cycle_total_cost * (1 + params['Profit_Margin']/100)) / total_lc_revenue_nm
    
    # Revenue CASM (per seat)
    revenue_casm = revenue_required_per_nm / params['Aircraft_PAX_Seats']
    
    # Fare per pax mile (assuming 5 pax which is typical for a thin haul route)
    passengers = 5
    fare_per_pax_nm = revenue_required_per_nm / passengers
    
    return LifeCycleResults(
        annual_costs=costs,
        time_series=t,
        cumulative_costs=y[1, :],
        life_cycle_total_cost=life_cycle_total_cost,
        revenue_casm=revenue_casm,
        fare_per_pax_nm=fare_per_pax_nm,
        pax_seats=params['Aircraft_PAX_Seats']
    )


def extract_aircraft_name(filename: str) -> str:
    """Extract a readable aircraft name from the filename."""
    name = Path(filename).stem
    name = name.replace('_Cost_Model_1500hrs', '')
    name = name.replace('_', ' ')
    return name


def run_comparison(file_list: List[str]) -> Tuple[pd.DataFrame, Dict[str, LifeCycleResults]]:
    """
    Run comparison across all aircraft configurations.
    
    Args:
        file_list: List of paths to Excel parameter files
        
    Returns:
        Tuple of (summary DataFrame, dict of full results by aircraft name)
    """
    results = []
    full_results = {}
    
    for file_path in file_list:
        try:
            aircraft_name = extract_aircraft_name(file_path)
            print(f"Processing: {aircraft_name}...")
            
            params = load_parameters(file_path)
            lc_results = calculate_life_cycle_results(params)
            
            # Extract summary metrics
            costs = lc_results.annual_costs
            summary = {
                'Aircraft': aircraft_name,
                'Revenue_CASM': lc_results.revenue_casm,
                'Fare_per_Pax_NM': lc_results.fare_per_pax_nm,
                'Total_Annual_Costs': costs.total_annual_costs,
                'Life_Cycle_Total_Cost': lc_results.life_cycle_total_cost,
                'Annual_Fixed_Costs': costs.annual_fixed_costs,
                'Annual_Variable_Costs': costs.annual_variable_costs,
                'Aircraft_Purchase_Price': costs.aircraft_purchase_price,
                'Variable_Cost_per_Hour': costs.total_variable_cost_per_hour,
                'Aircraft_Speed': costs.aircraft_speed,
                'Pilots_per_Aircraft': costs.pilots_per_aircraft,
                'PAX_Seats': lc_results.pax_seats,
            }
            results.append(summary)
            full_results[aircraft_name] = lc_results
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
    
    df = pd.DataFrame(results)
    
    cols = ['Aircraft', 'Revenue_CASM', 'Fare_per_Pax_NM', 'Total_Annual_Costs', 
            'Life_Cycle_Total_Cost', 'Annual_Fixed_Costs', 'Annual_Variable_Costs',
            'Aircraft_Purchase_Price', 'Variable_Cost_per_Hour', 'Aircraft_Speed', 
            'Pilots_per_Aircraft', 'PAX_Seats']
    df = df[[c for c in cols if c in df.columns]]
    
    return df, full_results


########################################
# Visualization and Output Functions
########################################

def create_comparison_charts(df: pd.DataFrame, full_results: Dict[str, LifeCycleResults], 
                            output_dir: Path):
    """Create comparison bar charts and cumulative cost plot."""
    df_sorted = df.sort_values('Revenue_CASM', ascending=True)
    
    # Figure 1: Revenue CASM and Fare per Passenger-Mile comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(df_sorted) * 0.8)))
    fig.suptitle('Aircraft Life Cycle Cost Comparison', fontsize=16, fontweight='bold', y=1.02)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df_sorted)))
    
    # Chart 1: Revenue CASM
    ax1 = axes[0]
    bars1 = ax1.barh(df_sorted['Aircraft'], df_sorted['Revenue_CASM'], color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Revenue CASM ($/seat-NM)', fontsize=11)
    ax1.set_title('Revenue CASM\n(Cost per Available Seat Nautical Mile)', fontsize=12, fontweight='bold')
    
    for bar, val in zip(bars1, df_sorted['Revenue_CASM']):
        ax1.text(val + 0.002, bar.get_y() + bar.get_height()/2, f'${val:.4f}', 
                va='center', ha='left', fontsize=9)
    
    ax1.set_xlim(0, df_sorted['Revenue_CASM'].max() * 1.25)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Chart 2: Fare per Pax mile (5 pax)
    ax2 = axes[1]
    bars2 = ax2.barh(df_sorted['Aircraft'], df_sorted['Fare_per_Pax_NM'], color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Fare per Pax-NM ($/pax-NM)', fontsize=11)
    ax2.set_title('Revenue per Pax-Mile\n(Assuming 5 Pax)', fontsize=12, fontweight='bold')
    
    for bar, val in zip(bars2, df_sorted['Fare_per_Pax_NM']):
        ax2.text(val + 0.002, bar.get_y() + bar.get_height()/2, f'${val:.4f}', 
                va='center', ha='left', fontsize=9)
    
    ax2.set_xlim(0, df_sorted['Fare_per_Pax_NM'].max() * 1.25)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    chart_path = output_dir / 'aircraft_comparison.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nComparison chart saved to: {chart_path}")
    
    # Figure 2: Cumulative costs over time
    fig2, ax3 = plt.subplots(figsize=(12, 8))
    ax3.set_title('Cumulative Life Cycle Costs Over Time', fontsize=14, fontweight='bold')
    
    color_map = plt.cm.tab10(np.linspace(0, 1, len(full_results)))
    
    for idx, (aircraft_name, lc_results) in enumerate(full_results.items()):
        t = lc_results.time_series
        cumulative = lc_results.cumulative_costs / 1e6  # Convert to millions
        ax3.plot(t, cumulative, label=aircraft_name, linewidth=2, color=color_map[idx])
    
    ax3.set_xlabel('Years', fontsize=12)
    ax3.set_ylabel('Cumulative Cost ($ Millions)', fontsize=12)
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim(0, None)
    ax3.set_ylim(0, None)
    
    plt.tight_layout()
    cumulative_path = output_dir / 'cumulative_costs.png'
    plt.savefig(cumulative_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Cumulative costs chart saved to: {cumulative_path}")
    
    # Figure 3: Pie charts for each aircraft's cost breakdown
    n_aircraft = len(full_results)
    if n_aircraft > 0:
        # Determine grid layout
        n_cols = min(3, n_aircraft)
        n_rows = (n_aircraft + n_cols - 1) // n_cols
        
        fig3, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        fig3.suptitle('Annual Cost Component Breakdown by Aircraft', fontsize=16, fontweight='bold', y=1.02)
        
        # Flatten axes for easy iteration
        if n_aircraft == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        # Define cost components for pie chart
        cost_labels = [
            'Amortization',
            'Depreciation', 
            'Insurance',
            'Personnel',
            'Training',
            'Facilities',
            'Fuel',
            'Maintenance',
            'Other Variable'
        ]
        
        pie_colors = plt.cm.Set3(np.linspace(0, 1, len(cost_labels)))
        
        for idx, (aircraft_name, lc_results) in enumerate(full_results.items()):
            ax = axes[idx]
            costs = lc_results.annual_costs
            
            # Group costs into categories
            insurance = costs.hull_insurance + costs.liability_insurance
            facilities = (costs.landing_site_annual_support + costs.hangar_and_office_expenses + 
                         costs.property_tax + costs.maintenance_software + costs.miscellaneous_service)
            
            # Calculate flight hours for annual variable cost breakdown
            flight_hours = (costs.annual_variable_costs / costs.total_variable_cost_per_hour 
                           if costs.total_variable_cost_per_hour > 0 else 0)
            
            maintenance_annual = (costs.maintenance_labor_per_hour + costs.schedule_parts_per_hour + 
                                  costs.midlife_inspection_per_hour + costs.engine_overhaul_per_hour +
                                  costs.propeller_allowance_per_hour) * flight_hours
            fuel_annual = costs.fuel_cost_per_hour * flight_hours
            other_variable = (costs.modernisation_per_hour + costs.paint_per_hour + 
                             costs.refurbishing_per_hour + costs.battery_per_hour +
                             costs.miscellaneous_trip_expenses_per_hour + costs.landing_fees_per_hour) * flight_hours
            
            cost_values = [
                costs.annual_amortization,
                costs.annual_depreciation,
                insurance,
                costs.annual_personnel_costs,
                costs.annual_training_cost,
                facilities,
                fuel_annual,
                maintenance_annual,
                other_variable
            ]
            
            # Filter out zero/negative values for cleaner pie chart
            filtered_labels = []
            filtered_values = []
            filtered_colors = []
            for label, value, color in zip(cost_labels, cost_values, pie_colors):
                if value > 0:
                    filtered_labels.append(label)
                    filtered_values.append(value)
                    filtered_colors.append(color)
            
            if filtered_values:
                wedges, texts, autotexts = ax.pie(
                    filtered_values, 
                    labels=filtered_labels,
                    autopct=lambda pct: f'{pct:.1f}%' if pct > 5 else '',
                    colors=filtered_colors,
                    startangle=90,
                    pctdistance=0.75,
                    wedgeprops={'edgecolor': 'white', 'linewidth': 1}
                )
                
                # Style the text
                for autotext in autotexts:
                    autotext.set_fontsize(8)
                    autotext.set_fontweight('bold')
                for text in texts:
                    text.set_fontsize(8)
                
                total = sum(filtered_values)
                ax.set_title(f'{aircraft_name}\n(${total:,.0f}/year)', fontsize=11, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No cost data', ha='center', va='center')
                ax.set_title(aircraft_name, fontsize=11, fontweight='bold')
        
        # Hide unused subplots
        for idx in range(n_aircraft, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        pie_path = output_dir / 'cost_breakdown_pies.png'
        plt.savefig(pie_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Cost breakdown pie charts saved to: {pie_path}")


def print_comparison_table(df: pd.DataFrame):
    """Print a formatted comparison table."""
    print("\n" + "=" * 105)
    print("AIRCRAFT COMPARISON SUMMARY")
    print("=" * 105)
    
    df_sorted = df.sort_values('Revenue_CASM', ascending=True)
    
    print(f"\n{'Aircraft':<25} {'Rev CASM':>12} {'$/Pax-NM':>12} {'Annual Cost':>15} {'LC Total':>18} {'Seats':>8}")
    print("-" * 105)
    
    for _, row in df_sorted.iterrows():
        print(f"{row['Aircraft']:<25} ${row['Revenue_CASM']:>10.4f} ${row['Fare_per_Pax_NM']:>10.4f} "
              f"${row['Total_Annual_Costs']:>13,.0f} ${row['Life_Cycle_Total_Cost']:>16,.0f} {row['PAX_Seats']:>7.0f}")
    
    print("-" * 105)
    print("\nNotes:")
    print("  - Rev CASM: Revenue Cost per Available Seat Nautical Mile (with profit margin)")
    print("  - $/Pax-NM: Revenue required per Passenger Nautical Mile (assuming 5 passengers)")
    print("  - Annual Cost: Total annual operating cost")
    print("  - LC Total: Life Cycle Total Cost")
    print("=" * 105)


def main(file_list: List[str] = None):
    """Find all Excel files and run comparison."""
    
    script_dir = Path(__file__).parent.resolve()
    
    # Find the excel files in the script directory
    excel_files = list(script_dir.glob('*Cost_Model*.xlsx'))
                
    print(f"\nFound {len(excel_files)} configuration file(s):")
    for f in excel_files:
        print(f"  - {f.name}")
    print()
    
    df, full_results = run_comparison([str(f) for f in excel_files])
    
    print_comparison_table(df)
    
    create_comparison_charts(df, full_results, script_dir)
    
    return df, full_results


if __name__ == "__main__":
    results = main()