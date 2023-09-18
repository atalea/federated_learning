# import numpy as np
# import matplotlib.pyplot as plt

# # Sample data for three arrays
# time_slots = ['Time Slot 1', 'Time Slot 2', 'Time Slot 3']
# arr1 = [10, 20, 30]
# arr2 = [12, 15, 40]
# arr3 = [99, 10, 20]

# # Plotting
# fig, ax1 = plt.subplots(figsize=(10, 5))

# # Plot Array 1 (using the first y-axis)
# ax1.plot(time_slots, arr1, 'b.-', label='Array 1')
# for i, val in enumerate(arr1):
#     ax1.text(time_slots[i], val, str(val),
#              ha='center', va='bottom', fontsize=9)

# # Plot Array 2 (using the first y-axis)
# ax1.plot(time_slots, arr2, 'g.-', label='Array 2')
# for i, val in enumerate(arr2):
#     ax1.text(time_slots[i], val, str(val),
#              ha='center', va='bottom', fontsize=9)

# ax1.set_xlabel('Time Slot')
# ax1.set_ylabel('Value (Array 1 & Array 2)')
# ax1.set_title('Arrays in Three Time Slots')

# # Plot Array 3 (using the second y-axis) as a line plot for percentage values
# ax2 = ax1.twinx()
# ax2.plot(time_slots, arr3, 'r.-', label='Array 3')
# for i, val in enumerate(arr3):
#     ax2.text(time_slots[i], val, f'{val}%',
#              ha='center', va='bottom', fontsize=9)

# # Set y-axis limit for better visualization of Array 3
# ax2.set_ylim(0, max(arr3)*1.2)  # Increase the limit slightly to add padding

# ax2.set_ylabel('Percentage (Array 3)', color='r')

# # Show legend for both y-axes
# ax1.legend(loc='upper left', bbox_to_anchor=(0.01, 0.98))
# ax2.legend(loc='upper right', bbox_to_anchor=(0.99, 0.98))

# plt.grid()

# plt.tight_layout()
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

fedavg_accu = [84, 88, 90, 91, 92, 93, 96]
fedavg_loss = [0.55, 0.50, 0.49, 0.44, 0.42, 0.39, 0.35]
fedavg_power = [10, 33, 45, 24, 35, 60, 89]
ibcs_accu = [79, 80, 84, 89, 92, 94, 98]
ibcs_loss = [0.60, 0.55, 0.49, 0.43, 0.39, 0.37, 0.32]
ibcs_power = [16, 17, 18, 34, 55, 86, 94]

fig, ax = plt.subplots()
ax.plot(fedavg_accu)
ax.plot(ibcs_accu)


ax.set_title('Accuracy')
ax.legend(['FedAvg', 'IBCS'])
ax.xaxis.set_label_text('Gobal Epochs')
ax.yaxis.set_label_text('Accuracy in %')

fig, ax = plt.subplots()
ax.plot(fedavg_loss)
ax.plot(ibcs_loss)


ax.set_title('Loss')
ax.legend(['FedAvg', 'IBCS'])
ax.xaxis.set_label_text('Gobal Epochs')
ax.yaxis.set_label_text('Loss')

fig, ax = plt.subplots()
ax.plot(fedavg_power)
ax.plot(ibcs_power)


ax.set_title('Power')
ax.legend(['FedAvg', 'IBCS'])
ax.xaxis.set_label_text('Gobal Epochs')
ax.yaxis.set_label_text('Power')
plt.show()

# fig, axs = plt.subplots(1, 3)
# axs[0].plot(fedavg_accu/100)
# axs[0].plot(ibcs_accu/100)
# axs[0].set_title('Accuracy')
# axs[0].set(xlabel='x-label', ylabel='y-label')

# axs[1].plot(fedavg_loss)
# axs[1].plot(ibcs_loss)
# axs[1].set_title('Loss')
# axs[1].set(xlabel='x-label', ylabel='y-label')

# axs[2].plot(fedavg_power/100)
# axs[2].plot(ibcs_power/100)
# axs[2].set_title('Power')
# axs[2].set(xlabel='x-label', ylabel='y-label')

# for ax in axs.flat:
#     ax.set(xlabel='x-label', ylabel='y-label')

# Hide x labels and tick labels for top plots and y ticks for right plots.
# for ax in axs.flat:
#     ax.label_outer()

# plt.legend(['fed', 'y = 2x', 'y = 3x'], loc='upper left')

plt.show()
