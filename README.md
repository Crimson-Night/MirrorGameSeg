# Instructions
096260 Project


Currently we have 10 graphs in our data set. The master agent is trained on the first 5.

To run the GUI and contribute your input as a learning session do the following:

1.  execute run_gui.py
    You will be able to apply your own segmentation to the first 5 graphs, one at the time.
    See the GUI's instruction in it's openning screen.
    
    You can run as many sessions as you like.
    
2.  When you're ready, execute train_master.py
    This will begin a learnning session when the master agent will process all of your
    previous applications.
    
3.  Now we can let the master do his own segmentation on one of the 10 graphs with the knowledge
    it aquired from your sessions, and those that came before that.
    
    exectue segmentation.py, and provide a number 1 - 10, stating which graph you wish
    to apply the segmentation to.
    Remember: The master is trained only on the first 5 graphs, the rest of them are a wild card.
    Also, when running the segmentation on a graph, the master is self learning with respects to
    his own decitions made during the segmentaion process.
    Check results.log to see what you got.
