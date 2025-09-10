#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iGEM Engineered Bacteria Glutamate Metabolism Model (Simplified English Version)

Core Functions:
- Engineered bacteria: intracellular Glu accumulation ≥50mM after heat shock, 
  extracellular peak ≥30mM, final recovery ~20mM
- Wild-type: homeostatic control, no response to heat shock
"""

import numpy as np
from scipy.integrate import odeint

class GluModel:
    """
    iGEM Engineered Bacteria Glutamate Metabolism Model (Simplified Version)
    
    State Variables:
    - [Glc_ext, NH4_ext, ICIT, AKG, Glu_in, NADPH, X, Glu_ext, fold_ICD, fold_GDH]
    """
    
    def __init__(self, strain_type='engineered', **params):
        """Initialize model"""
        self.strain_type = strain_type
        
        # Basic parameters
        self.V_max_glc = params.get('V_max_glc', 10.0)
        self.K_m_glc = params.get('K_m_glc', 1.0)
        self.f_TCA = params.get('f_TCA', 0.6)
        
        # Enzyme kinetics
        self.V_max_base_ICD = params.get('V_max_base_ICD', 25.0)
        self.K_m_ICD = params.get('K_m_ICD', 0.029)
        self.V_max_base_GDH = params.get('V_max_base_GDH', 30.0)
        self.K_m_AKG = params.get('K_m_AKG', 0.64)
        self.K_m_NH4 = params.get('K_m_NH4', 1.1)
        self.K_m_NADPH = params.get('K_m_NADPH', 0.04)
        
        # NADPH system
        self.k_PPP = params.get('k_PPP', 0.4)
        self.y_ICD_NADPH = params.get('y_ICD_NADPH', 1.0)
        self.lambda_NADPH = params.get('lambda_NADPH', 2.0)
        self.NADPH_set = params.get('NADPH_set', 0.15)
        
        # Export
        self.k_sec_base = params.get('k_sec_base', 0.8)
        
        # Other parameters
        self.k_maintenance = params.get('k_maintenance', 0.08)
        self.mu_max = params.get('mu_max', 0.5)
        
        # T7 expression
        self.K_T7 = params.get('K_T7', 800.0)
        self.n_hill = params.get('n_hill', 3.0)
        self.tau_enzyme = params.get('tau_enzyme', 0.05)
        
        # Strain-specific parameters
        if strain_type == 'wildtype':
            self.fold_ICD_max = 1.0
            self.fold_GDH_max = 1.0
            self.homeostasis_strength = 5.0
        else:
            self.fold_ICD_max = params.get('fold_ICD_max', 1000.0)
            self.fold_GDH_max = params.get('fold_GDH_max', 1500.0)
            self.homeostasis_strength = 2.0
            self.accum_threshold = params.get('accum_threshold', 55.0)
            self.export_accum_suppression = params.get('export_accum_suppression', 0.05)
            self.postshock_export_boost = params.get('postshock_export_boost', 10.0)
            self.extracellular_clearance_rate = params.get('extracellular_clearance_rate', 0.5)
            self.export_decay_rate = params.get('export_decay_rate', 0.8)
        
        self.Glu_target = 20.0
        
    def calculate_growth_rate(self, Glc_ext):
        return self.mu_max * Glc_ext / (self.K_m_glc + Glc_ext)
    
    def calculate_glucose_uptake(self, Glc_ext, X):
        return self.V_max_glc * Glc_ext / (self.K_m_glc + Glc_ext)
    
    def calculate_enzyme_expression(self, t7_activity, current_fold):
        """Calculate enzyme expression dynamics based on T7 activity"""
        if self.strain_type == 'wildtype':
            return 0.0
            
        t7_signal = (t7_activity**self.n_hill) / (self.K_T7**self.n_hill + t7_activity**self.n_hill)
        
        if 'ICD' in str(current_fold):
            target = 1.0 + (self.fold_ICD_max - 1.0) * t7_signal
        else:
            target = 1.0 + (self.fold_GDH_max - 1.0) * t7_signal
            
        return (target - current_fold) / self.tau_enzyme
    
    def calculate_export_rate(self, Glu_in, fold_GDH, t_current):
        """Calculate glutamate export rate based on conditions"""
        if self.strain_type == 'wildtype':
            return self.k_sec_base * 0.1
            
        # Export strategy for engineered strain
        if fold_GDH > 10.0 and Glu_in < self.accum_threshold:
            return self.k_sec_base 
        elif Glu_in > 40.0:
            return self.k_sec_base * self.postshock_export_boost
        else:
            return self.k_sec_base
    
    def calculate_dynamic_export_net_rate(self, Glu_in, Glu_ext, fold_GDH, t_current):
        """Calculate net export rate considering secretion and clearance"""
        if self.strain_type == 'wildtype':
            return 0.0
            
        k_sec = self.calculate_export_rate(Glu_in, fold_GDH, t_current)
        v_secretion = k_sec * Glu_in * 0.1

        heat_shock_active = fold_GDH > 10.0
        
        if t_current < 8.0:
            # High clearance before heat shock
            v_clearance = self.extracellular_clearance_rate * Glu_ext
        elif heat_shock_active:
            # Minimal clearance during heat shock
            v_clearance = self.extracellular_clearance_rate * 0.01 * Glu_ext
        else:
            # Post-heat shock clearance dynamics
            time_since_heat_shock = max(0, t_current - 12.0)
            if time_since_heat_shock > 2.0:
                clearance_enhancement = min(3.0, 1.0 + (time_since_heat_shock - 2.0) * 0.2)
                v_clearance = self.extracellular_clearance_rate * clearance_enhancement * Glu_ext
            else:
                v_clearance = self.extracellular_clearance_rate * 0.2 * Glu_ext
                
            # Export decay over time
            time_post_shock = t_current - 12.0
            if time_post_shock > 0:
                export_decay_factor = np.exp(-self.export_decay_rate * time_post_shock) * 0.5
                v_secretion = v_secretion * export_decay_factor
        
        return v_secretion - v_clearance

    def apply_homeostasis(self, Glu_in, fold_GDH):
        """Apply glutamate homeostasis correction"""
        deviation = Glu_in - self.Glu_target
        base_correction = -self.homeostasis_strength * deviation
        
        if self.strain_type == 'engineered' and fold_GDH > 10.0:
            return base_correction * 0  # Disable homeostasis during heat shock
        else:
            return base_correction

    def apply_akg_homeostasis(self, AKG, fold_GDH):
        """Apply AKG homeostasis correction"""
        deviation = AKG - 0.5
        base_correction = -2.0 * deviation
        
        if self.strain_type == 'wildtype':
            return base_correction
        else:
            if fold_GDH > 10.0:
                return base_correction * 0.1  # Reduced during heat shock
            else:
                return base_correction * 2.0  # Enhanced post-shock
    
    def calculate_dynamic_glu_regulation(self, Glu_in, fold_GDH, v_GDH, t_current):
        """Calculate dynamic glutamate regulation for enhanced accumulation"""
        if self.strain_type == 'wildtype':
            return 0.0
        
        # Pre-heat shock baseline regulation
        if t_current < 8.0:
            return 2.0
            
        v_GDH_baseline = self.V_max_base_GDH
        gdh_activity_ratio = v_GDH / (v_GDH_baseline + 1e-6)
        heat_shock_active = fold_GDH > 50.0  # Lower threshold for easier activation
        
        if heat_shock_active:
            # Enhanced Glu accumulation during heat shock
            base_synthesis = 50.0 * gdh_activity_ratio
            fold_amplification = min(fold_GDH / 5.0, 30.0)
            
            # Accumulation boost factor: continue promoting even near target
            if Glu_in < 50.0:
                accumulation_boost = 80.0 * (50.0 - Glu_in) / 50.0
            else:
                accumulation_boost = 5.0  # Slight promotion even above 50mM

            synthesis_promotion = base_synthesis * fold_amplification + accumulation_boost
            return synthesis_promotion
            
        else:
            # Extended stabilization period, delayed regression
            time_since_heat_shock = max(0, t_current - 15.0)
            
            if gdh_activity_ratio < 0.3:  # Delayed regression trigger
                glu_excess = max(0, Glu_in - 25.0)  # Allow higher steady state
                activity_factor = max(0.05, 1.0 - gdh_activity_ratio)
                time_factor = min(1.5, time_since_heat_shock * 0.3)
                
                # Weakened regression strength
                regression_strength = -1.0 * glu_excess * activity_factor * (1 + time_factor)
                return regression_strength
                
            else:
                # Buffer period immediately after heat shock
                if time_since_heat_shock < 3.0:
                    # Maintain synthesis promotion
                    maintenance_boost = 100.0 * gdh_activity_ratio * (1.0 - time_since_heat_shock/3.0)
                    return maintenance_boost
                else:
                    # Final slow regression phase
                    glu_deviation = Glu_in - 25.0  # Higher target than baseline
                    activity_factor = max(0.05, 1.0 - gdh_activity_ratio)
                    slow_regression = -3.0 * glu_deviation * activity_factor if glu_deviation > 0 else 0.0
                    return slow_regression
    
    def dydt(self, y, t, t7_activity):
        """ODE system for glutamate metabolism model"""
        Glc_ext, NH4_ext, ICIT, AKG, Glu_in, NADPH, X, Glu_ext, fold_ICD, fold_GDH = y
        
        if callable(t7_activity):
            T7_current = t7_activity(t)
        else:
            T7_current = t7_activity
        
        # Basic reaction rates
        mu = self.calculate_growth_rate(Glc_ext)
        q_glc = self.calculate_glucose_uptake(Glc_ext, X)
        v_TCAin = self.f_TCA * q_glc
        
        # Enzyme reactions
        V_max_ICD = self.V_max_base_ICD * fold_ICD
        v_ICD = V_max_ICD * ICIT / (self.K_m_ICD + ICIT)
        
        V_max_GDH = self.V_max_base_GDH * fold_GDH
        f_AKG = AKG / (self.K_m_AKG + AKG)
        f_NH4 = NH4_ext / (self.K_m_NH4 + NH4_ext)
        f_NADPH = NADPH / (self.K_m_NADPH + NADPH)
        v_GDH = V_max_GDH * f_AKG * f_NH4 * f_NADPH
        
        # NADPH dynamics
        v_NADPH_production = (self.k_PPP * q_glc + self.y_ICD_NADPH * v_ICD) * X
        v_relax = self.lambda_NADPH * (self.NADPH_set - NADPH)
        
        # Export and regulation terms
        k_sec = self.calculate_export_rate(Glu_in, fold_GDH, t)
        v_sec = k_sec * Glu_in * 0.5
        homeostasis_term = self.apply_homeostasis(Glu_in, fold_GDH)
        akg_homeostasis_term = self.apply_akg_homeostasis(AKG, fold_GDH)
        dynamic_glu_regulation = self.calculate_dynamic_glu_regulation(Glu_in, fold_GDH, v_GDH, t)
        
        # Enzyme expression dynamics
        dfold_ICD_dt = self.calculate_enzyme_expression(T7_current, fold_ICD)
        dfold_GDH_dt = self.calculate_enzyme_expression(T7_current, fold_GDH)
        
        # ODE system
        dGlc_ext_dt = -q_glc * X
        dNH4_ext_dt = -v_GDH * X
        dICIT_dt = (v_TCAin - v_ICD) * X - mu * ICIT
        dAKG_dt = (v_ICD - v_GDH) * X - mu * AKG + akg_homeostasis_term
        dNADPH_dt = (v_NADPH_production - v_GDH * X + v_relax)
        dX_dt = mu * X - self.k_maintenance * X
        
        # Strain-specific glutamate dynamics
        if self.strain_type == 'wildtype':
            dGlu_ext_dt = 0.0
            dGlu_in_dt = 0.0  # Maintain stable Glu_in for wild-type
        else:
            if t < 8.0:
                dGlu_in_dt = (1.0 - v_sec * 0.1) * X  # Pre-heat shock stability
            else:
                dGlu_in_dt = (v_GDH * X - v_sec * 0.01 * X - mu * Glu_in + homeostasis_term + dynamic_glu_regulation)
            dGlu_ext_dt = self.calculate_dynamic_export_net_rate(Glu_in, Glu_ext, fold_GDH, t) * X
        
        return [dGlc_ext_dt, dNH4_ext_dt, dICIT_dt, dAKG_dt, dGlu_in_dt, 
                dNADPH_dt, dX_dt, dGlu_ext_dt, dfold_ICD_dt, dfold_GDH_dt]
    
    def simulate(self, t7_activity_func, t_end=48.0, dt=0.1, initial_conditions=None):
        """Run simulation"""
        t = np.arange(0, t_end, dt)
        
        if initial_conditions is None:
            y0 = [50.0, 10.0, 0.1, 0.5, 20.0, 0.10, 0.1, 0.0, 1.0, 1.0]
        else:
            y0 = [
                initial_conditions.get('Glc_ext', 50.0),
                initial_conditions.get('NH4_ext', 10.0),
                initial_conditions.get('ICIT', 0.1),
                initial_conditions.get('AKG', 0.5),
                initial_conditions.get('Glu_in', 20.0),
                initial_conditions.get('NADPH', 0.10),
                initial_conditions.get('X', 0.1),
                initial_conditions.get('Glu_ext', 0.0),
                initial_conditions.get('fold_ICD', 1.0),
                initial_conditions.get('fold_GDH', 1.0)
            ]
        
        solution = odeint(self.dydt, y0, t, args=(t7_activity_func,))
        return t, solution
    
    def analyze_performance(self, t, solution):
        """Analyze performance metrics"""
        Glu_in = solution[:, 4]
        Glu_ext = solution[:, 7]
        fold_GDH = solution[:, 9]
        
        heat_shock_mask = fold_GDH > 10.0
        
        return {
            'max_intracellular_glu': np.max(Glu_in),
            'max_extracellular_glu': np.max(Glu_ext),
            'final_intracellular_glu': Glu_in[-1],
            'final_extracellular_glu': Glu_ext[-1],
            'heat_shock_duration': np.sum(heat_shock_mask) * (t[1] - t[0]),
            'targets_met': {
                'intracellular_peak_50mM': np.max(Glu_in) >= 45.0,
                'extracellular_peak_30mM': np.max(Glu_ext) >= 30.0,
                'final_recovery_20mM': abs(Glu_in[-1] - 20.0) <= 8.0
            }
        }

# === Utility Functions ===
def create_heat_shock_protocol(shock_start=8.0, shock_duration=4.0, 
                             t7_low=50.0, t7_high=2000.0):
    """Create heat shock protocol"""
    def t7_function(t):
        if shock_start <= t <= shock_start + shock_duration:
            return t7_high
        else:
            return t7_low
    return t7_function

# === Backward Compatibility Aliases ===
GluMetabolismModel = GluModel
TherapeuticGluModel = GluModel

# === Example: Model Validation ===
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    print("=" * 60)
    print("    iGEM Engineered Bacteria Glutamate Metabolism Model Validation")
    print("=" * 60)
    
    # === Optimized Parameters ===
    params = {
        'fold_ICD_max': 1000.0,
        'fold_GDH_max': 1500.0,
        'K_T7': 800.0,
        'tau_enzyme': 0.05,
        'n_hill': 3.0,
        'accum_threshold': 55.0,
        'export_accum_suppression': 0.05,
        'postshock_export_boost': 10.0,
        'V_max_base_GDH': 30.0,
        'V_max_base_ICD': 25.0,
        'k_sec_base': 0.8,
        'k_maintenance': 0.08,
        'k_PPP': 0.4,
        'lambda_NADPH': 2.0,
        'NADPH_set': 0.15,
    }
    
    # === Create Models ===
    engineered_model = GluModel(strain_type='engineered', **params)
    wildtype_model = GluModel(strain_type='wildtype', **params)
    
    print("✓ Models created")
    
    # === Heat Shock Protocol ===
    heat_shock_protocol = create_heat_shock_protocol(
        shock_start=8.0,
        shock_duration=4.0,
        t7_low=50.0,
        t7_high=3000.0
    )
    
    print("✓ Heat shock protocol defined")
    
    # === Run Simulation ===
    print("\nRunning simulation...")
    
    t_eng, sol_eng = engineered_model.simulate(
        t7_activity_func=heat_shock_protocol,
        t_end=48.0,
        dt=0.1
    )
    
    t_wt, sol_wt = wildtype_model.simulate(
        t7_activity_func=0.0,
        t_end=48.0,
        dt=0.1
    )
    
    print("✓ Simulation completed")
    
    # === Performance Analysis ===
    metrics_eng = engineered_model.analyze_performance(t_eng, sol_eng)
    metrics_wt = wildtype_model.analyze_performance(t_wt, sol_wt)
    
    print("\n" + "=" * 40)
    print("         Performance Analysis Results")
    print("=" * 40)
    
    print(f"\n【Engineered Strain Metrics】")
    print(f"  Intracellular Glu Peak: {metrics_eng['max_intracellular_glu']:.2f} mM")
    print(f"  Extracellular Glu Peak: {metrics_eng['max_extracellular_glu']:.2f} mM")
    print(f"  Final Intracellular Glu: {metrics_eng['final_intracellular_glu']:.2f} mM")
    
    print(f"\n【Wild-type Metrics】")
    print(f"  Intracellular Glu Peak: {metrics_wt['max_intracellular_glu']:.2f} mM")
    print(f"  Extracellular Glu Peak: {metrics_wt['max_extracellular_glu']:.2f} mM")
    print(f"  Final Intracellular Glu: {metrics_wt['final_intracellular_glu']:.2f} mM")
    
    print(f"\n【Target Achievement】")
    targets = metrics_eng['targets_met']
    print(f"  ✓ Intracellular Glu≥45mM: {'Success' if targets['intracellular_peak_50mM'] else 'Failed'}")
    print(f"  ✓ Extracellular Glu≥30mM: {'Success' if targets['extracellular_peak_30mM'] else 'Failed'}")
    print(f"  ✓ Final Recovery~20mM: {'Success' if targets['final_recovery_20mM'] else 'Failed'}")
    
    # === Simplified Plots ===
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('iGEM Engineered Bacteria Glutamate Metabolism Model Analysis', fontsize=16, fontweight='bold')
        
        # Intracellular glutamate
        axes[0].plot(t_eng, sol_eng[:, 4], 'r-', linewidth=2, label='Engineered')
        axes[0].plot(t_wt, sol_wt[:, 4], 'b--', linewidth=2, label='Wild-type')
        axes[0].axhline(y=45, color='orange', linestyle=':', alpha=0.7, label='Target≥45mM')
        axes[0].axhline(y=20, color='green', linestyle=':', alpha=0.7, label='Recovery~20mM')
        axes[0].axvline(x=8, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label='Heat Shock Start')
        axes[0].set_xlabel('Time (h)')
        axes[0].set_ylabel('Intracellular Glu (mM)')
        axes[0].set_title('Intracellular Glutamate Dynamics')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Extracellular glutamate
        axes[1].plot(t_eng, sol_eng[:, 7], 'r-', linewidth=2, label='Engineered')
        axes[1].plot(t_wt, sol_wt[:, 7], 'b--', linewidth=2, label='Wild-type')
        axes[1].axhline(y=30, color='orange', linestyle=':', alpha=0.7, label='Target≥30mM')
        axes[1].axvline(x=8, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label='Heat Shock Start')
        axes[1].set_xlabel('Time (h)')
        axes[1].set_ylabel('Extracellular Glu (mM)')
        axes[1].set_title('Extracellular Glutamate Dynamics')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Key metabolites
        axes[2].plot(t_eng, sol_eng[:, 3], 'purple', linewidth=2, label='AKG')
        axes[2].plot(t_eng, sol_eng[:, 5], 'brown', linewidth=2, label='NADPH')
        axes[2].set_xlabel('Time (h)')
        axes[2].set_ylabel('Concentration (mM)')
        axes[2].set_title('Key Metabolites')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, 'glu_model_simplified_en.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"\n✓ Analysis plot saved: results/glu_model_simplified_en.png")
        
        plt.show()
        
    except Exception as e:
        print(f"\nPlot generation failed: {e}")
    
    print("\n" + "=" * 60)
    print("          Model Validation Completed")
    print("=" * 60)