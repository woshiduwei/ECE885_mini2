# ECE885_mini2
This is the homework for MSU ECE885 mini project 2

ECE-CSE 885 2018
Mini-project#2
Due: Friday, March 16, 2018 , by 12 mid-nite (can be extended at a grade penalty (loss) of 5%/day for up to 5 days.
Exploring code(s)/model(s) for sRNN and LSTM RNN based on the BPTT algorithm (and/or any variants). The description here will focus on the KERAS 2.0 Library. Any equivalent examples from other libraries will be accepted.
Keras 2.0 Examples-- choose one of (Actual, we should divide the class into groups of equal membership)
(1) imdb_lstm.py
(2) imdb_cnn_lstm.py
(3) conv_lstm.py
(4) imdb_bidirectional_lstm.py
40%
1) Run your assigned example at default values. For you example, experiment with at least 2 other optimizers from Keras. Define your own (different) network and compare its performance with the assigned default example. Use reasonable common epochs to make the comparison. (You will receive full credit if your accuracy is no less than the default example, and would receive additional 1% for every 1% accuracy above the default example, up to 3%).
20%
2) Select one example from the 4 above. Use the “exponential learning rate” to create a modified version. Compare the performance with the original selected example using a roster or a plot under the same conditions including number of epochs
20% (10%+10%)
3) In the code (min-char-rnn.py) discussed in class,
replace:
dbh += dhraw
dWxh += np.dot(dhraw, xs[t].T)
dWhh += np.dot(dhraw, hs[t-1].T)
By
dbh += dh
dWxh += np.dot(dh, xs[t].T)
dWhh += np.dot(dh, hs[t-1].T)
10%
3.1) For the same conditions, compare the modified example performance (in terms of logloss), using the input.txt file, with the original code. Summarize your comparative results.
10%
3.2) Did the modified version worked or did not work for you. Explain precisely why. (no credit for partial, or incomplete answers).
10% (The challenge)
4) In the code (min-char-rnn.py) discussed in class,
Add the L1 norm or its smooth approximation to the existing loss function-- this requires you to modify the BPTT as well.) Compare the performance with the original code on the provided input.txt file.
A brief summary of your results should be in the form of a mini-report highlighting the key findings. Sample topics may include
-Summary description of each case
-Lessons learned: what worked and failed. What new things have you learned/discovered if any that you might share with me and/or the class.
-include your modified code with the report via Gitlab.
