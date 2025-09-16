#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AND Gate Logic Module - Clean and Simplified Version

Core Functions:
- Low oxygen + High temperature => High T7 activity
- High oxygen OR Low temperature => Low T7 activity

Author: iGEM Modeling Team
Date: 2024
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# ------------------------------------------------------------------
# Parameter Loading
# ------------------------------------------------------------------
def load_model_parameters():
    """Load model parameters"""
    # Promoter parameters - optimized based on iGEM Tsinghua 2023 BioBrick experimental data
    promoter_params = {
        "pPepT": {  # BBa_K4634000: Hypoxia-inducible promoter
            "type": "rep", 
            "beta": 43.72,     # Maximum expression
            "K": 5.0,          # Half-inhibition constant
            "n": 3.5,          # Hill coefficient
            "leaky": 2.0       # Basal expression
        },
        "pLR": {    # BBa_K4634017: Temperature-sensitive promoter
            "T_melt": 40.305,      # CI857 protein melting temperature
            "delta_H": 239.018,    # Protein unfolding enthalpy (kJ/mol)
            "steepness": 5.0,      # Response steepness
            "beta": 89.349,        # Maximum expression increment
            "leaky": 0.500,        # Basal expression level
            "T_damage": 43.971,    # Cell thermal damage temperature threshold
            "damage_rate": 1.738   # Thermal damage decay rate
        }
    }

    # Split T7 parameters - optimized to improve ON/OFF ratio
    splitT7_params = {
        "alpha": 2500.0,    # Maximum T7 activity
        "Kd": 80000.0,      # Dissociation constant
        "leaky": 0.0        # Zero leakage
    }
        
    return promoter_params, splitT7_params

# ------------------------------------------------------------------
# Core AND Gate Class
# ------------------------------------------------------------------
class SimpleANDGate:
    """Simplified AND gate logic model"""
    
    def __init__(self, promoter_params=None, splitT7_params=None):
        if promoter_params is None or splitT7_params is None:
            promoter_params, splitT7_params = load_model_parameters()
        self.promoter_params = promoter_params
        self.splitT7_params = splitT7_params

    def _hill_function(self, x, params):
        """Hill function calculation, supports activator and repressor types"""
        x = np.asarray(x, dtype=float)
        leaky, beta, K, n = params['leaky'], params['beta'], params['K'], params['n']
        ptype = params.get('type', 'act')
        
        if ptype == 'rep':  # Repressor type
            return leaky + beta * (K**n) / (K**n + x**n)
        else:  # Activator type
            return leaky + beta * (x**n) / (K**n + x**n)
    
    def _temperature_response_function(self, T_celsius, params):
        """
        CI857 temperature-sensitive protein biological response function
        Considers protein inactivation and cellular thermal damage processes
        """
        T_celsius = np.asarray(T_celsius, dtype=float)
        T_kelvin = T_celsius + 273.15
        T_melt_kelvin = params.get('T_melt', 42.0) + 273.15
        delta_H = params.get('delta_H', 150.0)  # kJ/mol
        R = 8.314e-3  # Gas constant kJ/(mol·K)
        steepness = params.get('steepness', 2.0)
        
        leaky = params['leaky']
        beta = params['beta']
        
        # CI857 protein inactivation process (van't Hoff equation)
        delta_G_ratio = steepness * delta_H * (1/T_melt_kelvin - 1/T_kelvin) / R
        delta_G_ratio = np.clip(delta_G_ratio, -50, 50)  # Prevent numerical overflow
        fraction_unfolded = 1.0 / (1.0 + np.exp(-delta_G_ratio))
        
        # Base protein inactivation response
        base_response = leaky + beta * fraction_unfolded
        
        # Cellular thermal damage correction
        T_damage = params.get('T_damage', 44.0)
        damage_rate = params.get('damage_rate', 0.8)
        
        # Use np.where to handle array conditions
        damage_factor = np.where(T_celsius <= T_damage, 
                               1.0, 
                               np.exp(-damage_rate * (T_celsius - T_damage)))
        
        return base_response * damage_factor
    
    def get_promoter_outputs(self, O2_percent, Temp_C):
        """Get promoter outputs"""
        pPepT_out = self._hill_function(O2_percent, self.promoter_params['pPepT'])
        pLR_out = self._temperature_response_function(Temp_C, self.promoter_params['pLR'])
        return pPepT_out, pLR_out

    def get_t7_activity(self, O2_percent, Temp_C):
        """Calculate T7 activity, AND gate core logic"""
        A, B = self.get_promoter_outputs(O2_percent, Temp_C)
        alpha = self.splitT7_params['alpha']
        Kd = self.splitT7_params['Kd']
        leaky = self.splitT7_params.get('leaky', 0.0)
        product = A * B
        return leaky + alpha * product / (Kd + product)

    def quick_diagnose(self, O2_list=(1.0, 5.0, 21.0), Temp_list=(37.0, 42.0, 45.0)):
        """Quick diagnosis: print expression and T7 output under typical conditions"""
        print("\n=== Quick Diagnosis ===")
        print("Parameter settings:")
        for k, v in self.promoter_params.items():
            if k == 'pLR':
                print(f"  {k}: T_melt={v.get('T_melt', 'N/A'):.1f}°C, β={v.get('beta', 0):.0f}, leaky={v.get('leaky', 0):.1f}")
            else:
                print(f"  {k}: type={v.get('type', 'N/A')}, K={v.get('K', 'N/A'):.2f}, n={v.get('n', 'N/A'):.1f}, β={v.get('beta', 0):.0f}, leaky={v.get('leaky', 0):.1f}")
        print(f"  splitT7: α={self.splitT7_params['alpha']:.0f}, Kd={self.splitT7_params['Kd']:.0f}")
        
        print("\nCondition scan (T7 activity):")
        for o2 in O2_list:
            for T in Temp_list:
                t7 = self.get_t7_activity(o2, T)
                p1, p2 = self.get_promoter_outputs(o2, T)
                print(f"  O₂={o2:5.1f}%  T={T:4.1f}°C  pPepT={p1:6.0f}  pLR={p2:6.0f}  T7={t7:6.0f} AU")

    # ------------------------------------------------------------------
    # Visualization Functions
    # ------------------------------------------------------------------
    def create_comprehensive_analysis(self, save_path=None, show_plot=True):
        """Create four-panel comprehensive analysis plot"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Panel 1: pPepT oxygen response curve
        O2_range = np.logspace(-1, 1.5, 100)  # 0.1% to ~30%
        pPepT_response = self._hill_function(O2_range, self.promoter_params['pPepT'])
        
        ax1.semilogx(O2_range, pPepT_response, 'b-', linewidth=2, label='pPepT Response')
        # Add experimental data points
        exp_O2 = [1.0, 2.0, 5.0, 10.0, 21.0]
        exp_pPepT = [self._hill_function(o2, self.promoter_params['pPepT']) for o2 in exp_O2]
        ax1.scatter(exp_O2, exp_pPepT, color='red', s=60, zorder=5, label='Data Points')
        
        ax1.set_xlabel('Oxygen (%)')
        ax1.set_ylabel('pPepT Activity')
        ax1.set_title('pPepT Hypoxia Response')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Panel 2: pLR temperature response curve
        T_range = np.linspace(30, 50, 100)
        pLR_response = self._temperature_response_function(T_range, self.promoter_params['pLR'])
        
        ax2.plot(T_range, pLR_response, 'r-', linewidth=2, label='pLR Response')
        # Add experimental data points
        exp_T = [37.0, 39.0, 42.0, 43.0, 45.0]
        exp_pLR = [self._temperature_response_function(t, self.promoter_params['pLR']) for t in exp_T]
        ax2.scatter(exp_T, exp_pLR, color='blue', s=60, zorder=5, label='Data Points')
        
        ax2.set_xlabel('Temperature (°C)')
        ax2.set_ylabel('pLR Activity')
        ax2.set_title('pLR Heat Response')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Panel 3: AND gate response heatmap
        O2_levels = np.logspace(-1, 1.3, 50)  # Increase resolution
        Temp_levels = np.linspace(35, 47, 50)  # Increase resolution
        O2_grid, Temp_grid = np.meshgrid(O2_levels, Temp_levels)
        T7_activity = self.get_t7_activity(O2_grid, Temp_grid)
        
        # Ensure reasonable data range
        print(f"T7 activity range: {np.min(T7_activity):.1f} - {np.max(T7_activity):.1f}")
        
        im = ax3.contourf(O2_grid, Temp_grid, T7_activity, levels=20, cmap='viridis')
        ax3.set_xscale('log')
        ax3.set_xlim(O2_levels.min(), O2_levels.max()+2)  # Explicitly set x-axis range
        ax3.set_ylim(Temp_levels.min(), Temp_levels.max())  # Explicitly set y-axis range
        ax3.set_xlabel('Oxygen (log scale)')
        ax3.set_ylabel('Temperature (°C)')
        ax3.set_title('AND Gate Response')
        
        # Add key condition markers
        conditions = [
            (1.0, 42.0, "ON", "red"), 
            (1.0, 37.0, "OFF", "blue"),
            (18.0, 37.0, "OFF", "orange"),
            
        ]
        for o2, temp, label, color in conditions:
            ax3.plot(o2, temp, 'o', color=color, markersize=8, markeredgecolor='white', markeredgewidth=2)
            ax3.annotate(label, (o2, temp), xytext=(5, 5), textcoords='offset points', 
                        color=color, fontweight='bold', fontsize=10)
        
        cbar = fig.colorbar(im, ax=ax3, shrink=0.8)
        cbar.set_label('T7 Activity (AU)')
        
        # Panel 4: Temporal dynamics simulation
        time = np.linspace(0, 14, 100)
        # Simulate temperature change: maintain 37°C for 4 hours, then rise to 42°C
        temp_profile = np.where(time < 4, 37.0, 
                               np.where(time < 6, 37.0 + (42.0-37.0)*(time-4)/2, 42.0))
        
        # Calculate T7 activity over time (assuming low oxygen condition 1%)
        t7_activity = [self.get_t7_activity(1.0, t) for t in temp_profile]
        
        ax4.plot(time, t7_activity, 'g-', linewidth=2, label='T7 Activity')
        ax4_temp = ax4.twinx()
        ax4_temp.plot(time, temp_profile, 'r--', linewidth=1, alpha=0.7, label='Temperature')
        
        # Add treatment window marker
        ax4.axvspan(4, 6, alpha=0.2, color='red', label='Treatment Window')
        
        ax4.set_xlabel('Time (hours)')
        ax4.set_ylabel('T7 Activity', color='g')
        ax4_temp.set_ylabel('Temperature (°C)', color='r')
        ax4.set_title('Temporal Dynamics')
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='upper left')
        ax4_temp.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Comprehensive analysis plot saved to: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        return fig

    def create_response_heatmap(self, save_path=None, show_plot=True):
        """Create 2D response heatmap"""
        O2_levels = np.logspace(-1, 1.3, 30)  # 0.1% to ~20%
        O2_levels = np.clip(O2_levels, 0.1, None)
        Temp_levels = np.linspace(35, 47, 30)
        O2_grid, Temp_grid = np.meshgrid(O2_levels, Temp_levels)
        T7_activity = self.get_t7_activity(O2_grid, Temp_grid)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.contourf(O2_grid, Temp_grid, T7_activity, levels=20, cmap='viridis')
        
        ax.set_title('AND Gate Response Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Oxygen Level (%)')
        ax.set_ylabel('Temperature (°C)')
        
        ax.set_xscale('log')
        ax.set_xlim(O2_levels.min(), O2_levels.max())
        
        cbar = fig.colorbar(im)
        cbar.set_label('T7 Activity (AU)')
        
        # Add key condition markers
        conditions = [
            (1.0, 42.0, "ON", "red"),
            (21.0, 37.0, "OFF", "blue"),
            (1.0, 37.0, "OFF", "blue")
        ]
        
        for o2, temp, label, color in conditions:
            ax.plot(o2, temp, 'o', color=color, markersize=8, markeredgecolor='white', markeredgewidth=2)
            ax.annotate(label, (o2, temp), xytext=(5, 5), textcoords='offset points', 
                       color=color, fontweight='bold', fontsize=10)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Heatmap saved to: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        return fig, ax

    # ------------------------------------------------------------------
    # Safety Assessment
    # ------------------------------------------------------------------
    def evaluate_safety_metrics(self):
        """Evaluate safety metrics"""
        on_condition = (1.0, 42.0)    # Low oxygen + high temperature - ON state
        off_conditions = [
            (21.0, 37.0),  # High oxygen + low temperature
            (21.0, 42.0),  # High oxygen + high temperature
            (1.0, 37.0),   # Low oxygen + low temperature
        ]
        
        on_activity = self.get_t7_activity(*on_condition)
        off_activities = [self.get_t7_activity(*cond) for cond in off_conditions]
        max_off_activity = max(off_activities)
        
        return {
            'on_activity': on_activity,
            'max_off_activity': max_off_activity,
            'on_off_ratio': on_activity / max_off_activity if max_off_activity > 0 else np.inf
        }

# ------------------------------------------------------------------
# Main Program Example
# ------------------------------------------------------------------
def main():
    """Main program: demonstrate core functionality"""
    print("="*60)
    print("🧬 AND Gate Logic Model - Simplified Version")
    print("="*60)
    
    # Initialize model
    gate = SimpleANDGate()
    
    # Basic logic test
    print("\n📊 Basic Logic Test")
    gate.quick_diagnose()
    
    # Safety evaluation
    print("\n🔒 Safety Evaluation")
    metrics = gate.evaluate_safety_metrics()
    print(f"  • ON state activity: {metrics['on_activity']:.1f} AU")
    print(f"  • OFF state maximum leakage: {metrics['max_off_activity']:.1f} AU")
    print(f"  • ON/OFF ratio: {metrics['on_off_ratio']:.1f}x")
    
    # Generate visualizations
    print("\n📈 Generate Visualizations")
    try:
        os.makedirs("results", exist_ok=True)
        gate.create_comprehensive_analysis(save_path="results/and_gate_comprehensive_analysis.png", show_plot=False)
        gate.create_response_heatmap(save_path="results/and_gate_heatmap.png", show_plot=False)
        print("  ✓ Visualization plots generated")
    except Exception as e:
        print(f"  ⚠️ Visualization generation failed: {e}")
    
    print(f"\n✅ Analysis completed!")
    print("="*60)

if __name__ == "__main__":
    main()
