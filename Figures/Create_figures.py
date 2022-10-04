import subprocess
from tqdm import tqdm

# this script will batch create all figures

program_list = ["/home/roberto/Github/De-Filippo-et-al-2022/Figures/Figure_1/Figure_1.py",
                "/home/roberto/Github/De-Filippo-et-al-2022/Figures/Figure_2/Figure_2.py",
                "/home/roberto/Github/De-Filippo-et-al-2022/Figures/Figure_3/Figure_3.py",
                "/home/roberto/Github/De-Filippo-et-al-2022/Figures/Figure_4/Figure_4.py",
                "/home/roberto/Github/De-Filippo-et-al-2022/Figures/Supplementary/Supplementary_Figure_probes_positions_areas.py",
                "/home/roberto/Github/De-Filippo-et-al-2022/Figures/Supplementary/Figure_1/Figure_1.py",
                "/home/roberto/Github/De-Filippo-et-al-2022/Figures/Supplementary/Supplementary_Figure_ML_sections_limits.py",
                "/home/roberto/Github/De-Filippo-et-al-2022/Figures/Supplementary/Figure_2/Figure_2.py",
                "/home/roberto/Github/De-Filippo-et-al-2022/Figures/Supplementary/Supplementary_Figure_strength_conservation_by_ripple_strength.py",
                "/home/roberto/Github/De-Filippo-et-al-2022/Figures/Supplementary/Supplementary_Figure_ripple_strength_across_ML.py",
                "/home/roberto/Github/De-Filippo-et-al-2022/Figures/Supplementary/Supplementary_Figure_ripple_amplitude_across_ML.py",
                "/home/roberto/Github/De-Filippo-et-al-2022/Figures/Supplementary/Supplementary_Figure_summary_heatmap_spike_differences_exc_inh.py",
                "/home/roberto/Github/De-Filippo-et-al-2022/Figures/Supplementary/Supplementary_Figure_spiking_rate_fraction_active_by_seed.py",
                "/home/roberto/Github/De-Filippo-et-al-2022/Figures/Supplementary/Supplementary_Figure_clusters_per_ripple_and_spiking_rate_early_late_exc_inh.py",
                "/home/roberto/Github/De-Filippo-et-al-2022/Figures/Supplementary/Supplementary_Figure_units_per_hippocampal_section.py"]

for program in tqdm(program_list):
    print(f"run {program}")
    subprocess.call(['python', program])
    print("Completed!")
