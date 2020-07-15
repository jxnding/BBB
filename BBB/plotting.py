def calibration_nn(nn_data, bnn_data, title='Calibration', output=''):
    import numpy as np
    import matplotlib.pyplot as plt
    import pdb
    #### Plotting
    fmt = 'o-'
    plt.plot([0,1], [0,1], 'k--') #theory line
    plt.errorbar(nn_data[2], nn_data[0], nn_data[1], capsize=5, fmt=fmt)
    plt.errorbar(bnn_data[2], bnn_data[0], bnn_data[1], capsize=5, fmt=fmt)
    plt.legend(['1:1', 'NN', 'BNN'])
    #### Title, labels
    # plt.ylim((0,1))
    # plt.xlim((0,1))
    plt.title(title)
    plt.xlabel('Output Probability of Classifier')
    plt.ylabel('Empirical Accuracy')
    #### Save and Close Fig
    plt.savefig(output+title+'.png', dpi=150)
    plt.close()

def calibration(gibbs_data, nn_data, bnn_data, title='Calibration', output=''):
    import numpy as np
    import matplotlib.pyplot as plt
    import pdb
    #### Plotting
    gibbs_acc = gibbs_data[3]
    fmt = 'o-'
    plt.plot([0,1], [0,1], 'k--') #theory line
    plt.errorbar(gibbs_data[0], gibbs_data[1], gibbs_data[2], capsize=5, fmt=fmt) # Gibbs
    plt.errorbar(nn_data[2], nn_data[0], nn_data[1], capsize=5, fmt=fmt)
    plt.errorbar(bnn_data[2], bnn_data[0], bnn_data[1], capsize=5, fmt=fmt)
    plt.legend(['1:1', 'Gibbs, Acc: '+str(gibbs_acc[0]/gibbs_acc[1]), 'NN', 'BNN'])
    #### Title, labels
    # plt.ylim((0,1))
    # plt.xlim((0,1))
    plt.title(title)
    plt.xlabel('Output Probability of Classifier')
    plt.ylabel('Empirical Accuracy')
    #### Save and Close Fig
    plt.savefig(output+title+'.png', dpi=150)
    plt.close()

def calibration_2(gibbs_data=[], gibbs_exact=[], nn_data=[], bnn_data=[], title='Calibration', output=''):
    import numpy as np
    import matplotlib.pyplot as plt
    import pdb
    #### Plotting
    fmt = 'o-'
    plt.plot([0,1], [0,1], 'k--') #theory line
    legend = ['1:1']
    if len(gibbs_data) > 0:
        gibbs_acc = gibbs_data[3]
        legend.append('Gibbs, Acc: '+str(gibbs_acc[0]/gibbs_acc[1]))
        plt.errorbar(gibbs_data[0], gibbs_data[1], gibbs_data[2], capsize=5, fmt=fmt) # Gibbs
    if len(nn_data) > 0:
        legend.append('NN')
        plt.errorbar(nn_data[2], nn_data[0], nn_data[1], capsize=5, fmt=fmt)
    if len(bnn_data) > 0:
        legend.append('BNN')
        plt.errorbar(bnn_data[2], bnn_data[0], bnn_data[1], capsize=5, fmt=fmt)
    if len(gibbs_exact) > 0:
        legend.append('Exact')
        plt.errorbar(gibbs_exact[0], gibbs_exact[1], gibbs_exact[2], capsize=5, fmt='o--', alpha=0.4) #, color='b'
    #### Title, labels
    # plt.ylim((0,1))
    # plt.xlim((0,1))
    plt.legend(legend)
    plt.title(title)
    plt.xlabel('Output Probability of Classifier')
    plt.ylabel('Empirical Accuracy')
    #### Save and Close Fig
    plt.savefig(output+title+'.png', dpi=150)
    plt.close()

def calibration_new(gibbs_data=[], gibbs_exact=[], nn_data=[], bnn_data=[], title='Calibration', output=''):
    import numpy as np
    import matplotlib.pyplot as plt
    import pdb
    #### Plotting
    fmt = 'o-'
    plt.plot([0,1], [0,1], 'k--') #theory line
    legend = ['1:1']
    if len(gibbs_data) > 0:
        gibbs_acc = gibbs_data[3]
        legend.append('Gibbs, '+str(gibbs_acc[0]/gibbs_acc[1]))
        plt.errorbar(gibbs_data[0], gibbs_data[1], gibbs_data[2], capsize=5, fmt=fmt) # Gibbs
    if len(nn_data) > 0:
        legend.append('NN, Acc: %2.2f, ECE: %2.2f' % (nn_data[-1], nn_data[-2]) )
        plt.errorbar(nn_data[0], nn_data[1], nn_data[2], capsize=5, fmt=fmt)
    if len(bnn_data) > 0:
        legend.append('BNN, Acc: %2.2f, ECE: %2.2f' % (bnn_data[-1], bnn_data[-2]) )
        plt.errorbar(bnn_data[0], bnn_data[1], bnn_data[2], capsize=5, fmt=fmt)
    if len(gibbs_exact) > 0:
        exact_acc = gibbs_exact[-1]
        legend.append('Exact, Acc: %2.2f, ECE: %2.2f' % ((exact_acc[0]/exact_acc[1]), gibbs_exact[-2]) )
        plt.errorbar(gibbs_exact[0], gibbs_exact[1], gibbs_exact[2], capsize=5, fmt=fmt, alpha=0.4, c='k') #, color='b'
    #### Title, labels
    # plt.ylim((0,1))
    # plt.xlim((0,1))
    plt.legend(legend)
    plt.title(title)
    plt.xlabel('Output Probability of Classifier')
    plt.ylabel('Empirical Accuracy')
    #### Save and Close Fig
    plt.savefig(output+title+'.png', dpi=150)
    plt.close()


def calibration_example(gibbs_data=[], gibbs_exact=[], nn_data=[], bnn_data=[], title='Calibration Example', output=''):
    import numpy as np
    import matplotlib.pyplot as plt
    import pdb
    x = np.arange(0,1.1,0.1)
    nn_data = np.arange(0, 0.55, .05)
    bnn_data = np.arange(0, 1.2, .2)
    #### Plotting
    fmt = 'o-'
    plt.plot([0,1], [0,1], 'k--') #theory line
    legend = ['1:1']
    if len(gibbs_data) > 0:
        gibbs_acc = gibbs_data[3]
        legend.append('Gibbs, '+str(gibbs_acc[0]/gibbs_acc[1]))
        plt.errorbar(gibbs_data[0], gibbs_data[1], gibbs_data[2], capsize=5, fmt=fmt) # Gibbs
    if len(nn_data) > 0:
        legend.append('Overconfident')
        plt.errorbar(x, nn_data, 0, capsize=5, fmt=fmt)
    if len(bnn_data) > 0:
        legend.append('Underconfident')
        plt.errorbar(np.arange(0,.6,0.1), bnn_data, 0, capsize=5, fmt=fmt)
    if len(gibbs_exact) > 0:
        exact_acc = gibbs_exact[-1]
        legend.append('Exact, Acc: %2.2f, ECE: %2.2f' % ((exact_acc[0]/exact_acc[1]), gibbs_exact[-2]) )
        plt.errorbar(gibbs_exact[0], gibbs_exact[1], gibbs_exact[2], capsize=5, fmt=fmt, alpha=0.4, c='k') #, color='b'
    #### Title, labels
    # plt.ylim((0,1))
    # plt.xlim((0,1))
    plt.legend(legend)
    plt.title(title)
    plt.xlabel('Output Probability of Classifier')
    plt.ylabel('Empirical Accuracy')
    #### Save and Close Fig
    plt.savefig(output+title+'.png', dpi=150)
    plt.close()
if __name__ == '__main__':
    calibration_example()