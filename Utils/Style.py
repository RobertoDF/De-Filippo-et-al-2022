import seaborn as sns


sns.set_theme(context='paper', style="ticks", rc={'legend.frameon': False, "legend.fontsize": 6, "axes.labelpad": 1 , "legend.title_fontsize": 6, 'axes.spines.right': False, 'axes.spines.top': False, "lines.linewidth": 0.5, "xtick.labelsize": 5, "ytick.labelsize": 5
                                                  , "xtick.major.pad": 2, "ytick.major.pad":2, "axes.labelsize": 6, "xtick.major.size": 1.5, "ytick.major.size": 1.5, "axes.titlesize" : 6 })


palette_figure_1 = {'CA1': "#301A4B",'DG': "#087E8B", 'CA2': "#F59A8C", 'CA3':"#5FAD56", 'SUB':"#AFA2FF",'ProS':"#FAC748"}

palette_timelags ={"Strong ripples":"#F03A47", "Long ripples":"#F03A47", "Total ripples": "#505A5B", "Common ripples": "#507DBC", "Local": "#00A676", "Non-local": "#6E2594"}

color_palette = sns.color_palette("flare", 255)

palette_ML = {'Medial seed': color_palette[0], "Lateral seed": color_palette[254], 'Central seed':  color_palette[int(254/2)], 'Medial': color_palette[0], 'Central':  color_palette[int(254/2)], "Lateral": color_palette[254]}

palette_type_neuron={"Putative exc": "#D64933", "Putative inh": "#00C2D1"}