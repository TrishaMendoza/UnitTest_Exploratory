from scipy import stats
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import Modules.firing_rate as firing_rate
import Modules.interspike_interval as interspike_interval
from Modules.Schriebber_Correlation import calculate_correlation
from Modules.load_data import spike_train_true, spike_train_predicted

# Set the seaborn theme 
sns.set_theme()

# Plot a sample of the data first
sampling_rate = 1000
time_vector = np.arange(1,len(spike_train_true)+1) #/sampling_rate


#sns.lineplot(x = time_vector, y = spike_train_true[:,2], label = 'True')
#sns.lineplot(x = time_vector, y = spike_train_predicted[:,2], label = 'Predicted')
#plt.show()

# --------------------------------------------------------------------------------------------------------
# 1. Send to calculate the Spike Rate - then compare
spike_rate_true = firing_rate.firing_rate(spike_train_true, 500, 0.5, sampling_rate )
spike_rate_pred = firing_rate.firing_rate(spike_train_predicted, 500, 0.5, sampling_rate )

mean_rate_true = np.mean(spike_rate_true, axis = 0)*1000
mean_rate_pred = np.mean(spike_rate_pred, axis = 0)*1000
rate_p_val = np.zeros(3)


# Plot the Histograms
for ch in range(spike_train_true.shape[1]):

    statistic , p_value = stats.wilcoxon(spike_rate_true[:, ch], spike_rate_pred[:, ch])
    rate_p_val[ch] = np.round(p_value,4)

    sns.histplot(spike_rate_true[:, ch], kde =True, bins = 5, color = '#7BAFD4' , label = 'True')
    sns.histplot(spike_rate_pred[:, ch], kde = True,  bins = 5, color = '#F38BA0' , label = 'Predicted')
    plt.title(f"Wilcoxon P-Value: {p_value}")
    plt.show()


# --------------------------------------------------------------------------------------------------------
# 2. Send to calculate the interspike interval - then compared
isi_dict_true  = interspike_interval.interspike_interval(spike_train_true, sampling_rate)
isi_dict_pred  = interspike_interval.interspike_interval(spike_train_predicted, sampling_rate)

isi_mean_true = np.zeros(3)
isi_mean_pred = np.zeros(3)
isi_p_value = np.zeros(3) 
indx = 0

for key in isi_dict_true:
    statistic , p_value = stats.mannwhitneyu(isi_dict_true[key], isi_dict_pred[key], alternative = "two-sided")

    isi_mean_true[indx] = np.mean(isi_dict_true[key])
    isi_mean_pred[indx] = np.mean(isi_dict_pred[key])
    isi_p_value[indx] = np.round(p_value, 4)

    sns.histplot(isi_dict_true[key], kde =True, bins = 5, color = '#7BAFD4' , label = 'True')
    sns.histplot(isi_dict_pred[key], kde = True, bins = 5, color = '#F38BA0' , label = 'Predicted')
    plt.title(f"Mann Whitney P-Value: {p_value}")
    plt.tight_layout()
    plt.savefig(f"plots/ch{indx}.png")
    plt.show()

    indx = indx+1


# --------------------------------------------------------------------------------------------------------
# 3. Send to Schrieber Correlation Plot

# Testing a range of plots + intialize
sigma_vals = np.arange(1, 100, 1) / 1000
correlation_vals = np.zeros([len(sigma_vals), spike_train_true.shape[1]])
mean_corr = np.zeros(spike_train_true.shape[1])

for ch in range(spike_train_true.shape[1]):
    for indx, sigma in enumerate(sigma_vals):
        corr_val = calculate_correlation(spike_train_true[:,ch], spike_train_predicted[:,ch], sigma, sampling_rate)
        correlation_vals[indx, ch] = np.round(corr_val, 4)
    
    mean_corr[ch] = np.mean(correlation_vals[:, ch])
    sns.lineplot(x = sigma_vals, y = correlation_vals[:, ch], color = '#7BAFD4')
    plt.title(f"Mean Correlation: {mean_corr[ch]}")
    plt.xlabel('Sigma Val')
    plt.ylabel('Correlation Value [r]')
    plt.ylim(0,1.2)
    plt.show()

# --------------------------------------------------------------------------------------------------------
# 4. Develop a report
output_path = "spike_train_report.html"

with open(output_path, "w") as f:
    f.write("<html><head><title>Spike Train Report</title></head><body>")
    f.write("<h1>Spike Train Model Validation Report</h1>")
    f.write("<table border='1' cellpadding='5'>")

    f.write("<tr><th>Channel</th><th>Firing Rate(T vs P)</th><th>P-Value</th><th>ISI Mean(T vs P)</th><th>P-Value</th><th>Correlation</th><th>Status</th><th>Details</th></tr>")

    for ch in range(spike_rate_true.shape[1]):

        if ch == 0:
            status = "✅ Pass"
        else:
            status = "❌ Fail"


        plot_link = f"<a href='plots/ch{ch}.png'>View Plot</a>"
        f.write(f"<tr><td>{ch}</td>")
        f.write(f"<td>{mean_rate_true[ch]:.2f} vs {mean_rate_pred[ch]:.2f}*10-3</td>")


        if rate_p_val[ch] < 0.05:
            f.write(f"<td><0.05</td>")
        else:
            f.write(f"<td>{rate_p_val[ch]}</td>")

        f.write(f"<td>{isi_mean_true[ch]:.2f} vs {isi_mean_pred[ch]:.2f}</td>")

        if isi_p_value[ch] < 0.05:
            f.write(f"<td><0.05</td>")
        else:
            f.write(f"<td>{isi_p_value[ch]}</td>")

        f.write(f"<td>{mean_corr[ch]}</td>")  
        f.write(f"<td>{status}</td>")     
        f.write(f"<td>{plot_link}</td></tr>")
        


    



    





    








