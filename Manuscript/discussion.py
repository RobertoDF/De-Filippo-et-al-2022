from Utils.Results_variables import r_common, r_strong

discussion = "Our work shows for the first time that strong ripples propagates differentially along the hippocampal " \
             "longitudinal axis. This propagation idiosyncrasy can be explained by a specific ability of the hippocampal septal (medial) pole to produce longer " \
             "ripples that better entrain the hippocampal network and spread across the longitudinal axis. " \
             "It was previously observed that ripples located at the septal and temporal pole are generated indipendently from each other, " \
             "in addition, despite the presence of connections within the hippocampal longitudinal axis {Witter, 2007 #1155; van Strien, 2009 #1156}, " \
             "in the vast majority of cases ripples do not propagate to the opposite pole {Sosa, 2020 #1154}. "\
             "In accordance with these results, " \
             "we observed a strong effect of spatial distance on ripple power correlation confirming a previous study {Nitzan, 2022 #1157}: " \
             "the power correlation, predictably, was higher in CA1 pairs" \
             "closer to each other. The effect of distance was also apparent on the ripple chance of propagation, only half of the ripples generated in " \
             "the septal pole were detected additionally in the intermediate hippocampus (corresponding to the lateral section in our analysis). " \
             "This chance is much higher compared to the one reported regarding propagation between " \
             "opposite poles (~3.7%), it would be interesting to understand whether the temporal pole is also able to entrain the intermediate hippocampus" \
             "in similar fashion or it is a peculiarity of the septal pole. " \
             "Notably, the dataset we employed is limited to the septal and intermediate hippocampus.  \n" \
             "Ripples can arise at any location along the hippocampal longitudinal axis {Patel, 2013 #1133}, our analysis shows that ripples are, however, " \
             "not homogeneously generated across space. We observed important differences between strong ripples and common ripples generation. " \
             "Common ripples generation followed a gradient with higher likelihood in the intermediate section and lowest in the septal pole." \
             "Strong ripples, on the other hand, are mostly generated locally, i.e., a strong ripple detected in the medial section is most likely " \
             "generated in the medial section itself. Furthermore, only rarely a strong ripple generated in the intermediate hippocampus is able to propagate " \
             "towards the septal pole retaining its strong status (top 10%). Conversely strong ripples generated in the septal pole have a sizable chance of " \
             "propagate longitudinally and still be in the top 10% in terms of ripple strength. Ripples generated in the septal pole and in the " \
             "intermediate hippocampus have a significantly different ability of engaging the hippocampal networks in the 50-120 ms window " \
             "post ripple start, ripples generated in the septal pole activate more neurons, both excitatory and inhibitory, and, moreover, " \
             "can elicit an higher spiking rate per neuron. This is reflected by the fact that the position on the longitudinal axis explains " \
             f"{r_common*100}% and {r_strong*100}% of the variability in ripple duration in common and strong ripples respectively. Consistenly, the" \
             "temporal hippocampus has been shown to produce shorter ripples booth in awake and sleep {Sosa, 2020 #1154}." \
             "Long duration ripples has been shown to be of particular importance in situations of " \
             "high-memory demand {Fernández-Ruiz, 2019 #1121}, previous studies highlighted " \
             "the role of septal hippocampus in memory tasks and information processing {Hock, 1998 #1122;Moser, 1993 #1123;Moser, 1995 #1124;Steffenach, 2005 #1128;" \
             "Kheirbek, 2013 #1126;McGlinchey, 2018 #1125;Fanselow, 2010 #1129;Maras, 2014 #1130;Bradfield, 2020 #1131;Qin, 2020 #1132}."\
             "Ripple generation seems to be homogeneous across the hippocampal longitudinal axis {Patel, 2013 #1133}. " \
             "Our results can explain the specific role of septal hippocampus in memory tasks with its ability of generating particularly long" \
             "ripples that are able to strongly engage networks in the entire top half of the hippocampal formation for an extended time. What " \
             "is the reason that enables the septal pole to generate longer ripples? We can speculate that the different neuromodulatory input might play a role. " \
             ""