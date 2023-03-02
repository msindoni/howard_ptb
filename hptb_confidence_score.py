import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Take CSV file and convert it to a dataframe
def make_df(filename):
    df = pd.read_csv(filename)
    return df 

#converting dataframe into an array
def make_array(input_dataframe):
    data_array = np.array(input_dataframe) 
    return data_array

#converting array win/losses to 1 and -1
def win_loss_convert(input_array):
    data_array_2 = input_array[:, 1:]
    opponent_weeks = len(data_array_2[:, 0])
    num_times_hptb = len(data_array_2[0, :])
    
    num_only_array = np.zeros((opponent_weeks, num_times_hptb), dtype = object) #create array to add wins/loss to numbers 1 and 2
    num_only_list = [] #create an empty list
    for i in range(opponent_weeks): #go through each opponent
        for j in range(num_times_hptb): #go through each HBTP for that specific opponent
            win_loss = data_array_2[i, j] #selecting a specific pick for a specific opponent from the original array
            if win_loss == 'WIN': #turning 'WIN' into 1
                win_loss = 1
            elif win_loss == 'LOSS': #turning losses into 2
                win_loss = -1
            else:
                win_loss = data_array_2[i, j] #any item that is a score, keeping it the same
            num_only_list.append(win_loss) #adding win_loss for that specific week/opponent variable to a list
        num_only_array[i] = num_only_list #making 1st dimension of the empty array the previously made list
        num_only_list.clear() #clearing list for next opponent iteration
    return num_only_array


    #creating a confidence score for each pick
def confidence_score(num_only_array, data_array):
    confidence_list = []
    max_score_possible_normalized = []
    
    win_loss_list = []
    for i in range(len(num_only_array[:])): #isolating lists that are just picks, not the score of a game played
        for j in range(len(num_only_array[1])):
            current_pick = num_only_array[i, j]
            if current_pick == 1 or current_pick == -1:
                win_loss_list.append(current_pick)
            else:
                break
        
        #getting absolute confidence scores 
        confidence_score_list = []
        num_of_picks = len(win_loss_list)
        for i in range(len(win_loss_list)):
            weekly_score = i / num_of_picks * win_loss_list[i] #the win/loss get weighted by how many picks came before
            confidence_score_list.append(weekly_score)
        confidence_score = sum(confidence_score_list) #list gets summed to ge the final confidence score
        confidence_list.append(confidence_score)
        win_loss_list.clear()
    return(confidence_list)

#getting season normalized confidence scores  
def season_norm_cs (confidence_list):
    season_normalized_confidence_list = []
    max_score = max(confidence_list)
    for i in (confidence_list):
        normalized_score = i / max_score
        season_normalized_confidence_list.append(normalized_score)
    return season_normalized_confidence_list

#getting max score possible normalized confidence scores
def max_possible_norm_cs(num_only_array, data_array):
    confidence_list = []
    max_score_possible_normalized = []
    
    win_loss_list = []
    for i in range(len(num_only_array[:])): #isolating lists that are just picks, not the score of a game played
        for j in range(len(num_only_array[1])):
            current_pick = num_only_array[i, j]
            if current_pick == 1 or current_pick == -1:
                win_loss_list.append(current_pick)
            else:
                break
        
        #getting absolute confidence scores 
        confidence_score_list = []
        num_of_picks = len(win_loss_list)
        for k in range(len(win_loss_list)):
            k = k + 1
            weekly_score = k / num_of_picks * win_loss_list[k - 1] #the win/loss get weighted by how many picks came before
            confidence_score_list.append(weekly_score)
        confidence_score = sum(confidence_score_list) #list gets summed to ge the final confidence score
        confidence_list.append(confidence_score)

        #getting the max score possible for each game based on the number of picks beforehand   
        num_of_picks = len(win_loss_list)
        current_max_possible = []
        for j in range(num_of_picks):
            j = j + 1
            max_possible_score = j / num_of_picks #essentially getting the weight for each week with the final = 1
            current_max_possible.append(max_possible_score)
            
            #determining the actual norm score by dividing the calc score by the theoretical max
        max_score_possible = sum(current_max_possible)
        final_score = (confidence_score / max_score_possible) * 100 
        max_score_possible_normalized.append(final_score)
            
        #clearing the lists made in the loop
        confidence_score_list.clear()
        current_max_possible.clear()
        win_loss_list.clear()
    return max_score_possible_normalized

def make_plot(data_array, num_only_array, chosen_analysis):
    #generating actual win/loss from original data (not manually)
    team_week = data_array[:, 0]
    win_loss_list = []
    for i in range(len(num_only_array[:])): 
        for j in range(len(num_only_array[1])):
            current_pick = num_only_array[i, j]
            if current_pick == 1 or current_pick == -1:
                continue
            else:
                split_pick = current_pick.split()
                win_loss_list.append(split_pick[1])
                break
                
    #creating the dataframe to graph
    data = {'team_week': team_week,
           'confidence_score': chosen_analysis,
           'bills_win_or_loss': win_loss_list}
    df = pd.DataFrame(data, columns = ['team_week', 'confidence_score', 'bills_win_or_loss'])
    
    #generating the graph of normalized confidence scores
    palette = {'Win':"blue",
               'Loss':"red"}
    sns.barplot( x = "team_week", y = "confidence_score", data = df, hue = "bills_win_or_loss", palette = palette)
    plt.xticks(rotation = 90, fontsize = 10)
    plt.xlabel("Week/Opponent", fontsize = 10)
    plt.ylabel("Confidence Interval", fontsize = 10)
    plt.legend(title = "Bill's Win or Loss")
    plt.tight_layout()
    plt.show()

####################################################################################################

####################################################################################################
df = make_df('HPTB_2020.csv')
data_array = make_array(df)
num_only_array = win_loss_convert(data_array)
confidence_list = confidence_score(num_only_array, data_array)
season_norm_cs = season_norm_cs(confidence_list)
max_norm_cs = max_possible_norm_cs(num_only_array, data_array)

make_plot(data_array, num_only_array, confidence_list)