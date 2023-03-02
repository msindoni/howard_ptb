import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re

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
    data_array_2 = input_array[:, 1:]#gets rid of first item which is the week and opponent
    opponent_weeks = len(data_array_2[:, 0])
    num_times_hptb = len(data_array_2[0, :])
    
    num_only_array = np.zeros((opponent_weeks, num_times_hptb), dtype = object) #create array to add wins/loss to numbers 1 and -1
    num_only_list = [] #create an empty list
    for i in range(opponent_weeks): #go through each opponent
        for j in range(num_times_hptb): #go through each HBTP for that specific opponent
            win_loss = data_array_2[i, j] #selecting a specific pick for a specific opponent from the original array
            if win_loss == 'WIN': #turning 'WIN' into 1
                win_loss = 1
            elif win_loss == 'LOSS': #turning losses into -1
                win_loss = -1
            else:
                win_loss = data_array_2[i, j] #any item that is a score, keeping it the same
            num_only_list.append(win_loss) #adding win_loss for that specific week/opponent variable to a list
        num_only_array[i] = num_only_list #making 1st dimension of the empty array the previously made list
        num_only_list.clear() #clearing list for next opponent iteration
    return num_only_array

#creating a confidence score and max possible score for each pick
def confidence_score(num_only_array):
    confidence_list = []
    max_score_possible_list = []
    for i in range(len(num_only_array[:])): #isolating lists that are just picks, not the score of a game played
        win_loss_list = []
        for j in range(len(num_only_array[1])):
            current_pick = num_only_array[i, j]
            if current_pick == 1 or current_pick == -1:
                win_loss_list.append(current_pick)
            else:
                continue
        
        #getting absolute confidence and max confidence score possible for that week 
        confidence_score_list = []
        max_value_list = []
        num_of_picks = len(win_loss_list)
        for i in range(len(win_loss_list)):
            weekly_score = i / num_of_picks * win_loss_list[i] #the win/loss get weighted by how many picks came before
            max_value = i / num_of_picks
            confidence_score_list.append(weekly_score)
            max_value_list.append(max_value)
        confidence_list.append(sum(confidence_score_list))#list gets summed to ge the final confidence score
        max_score_possible_list.append(sum(max_value_list))
        print(confidence_list)
    print(confidence_list)
    return(confidence_list, max_score_possible_list)

def confidence_score_week_before(num_only_array):
    confidence_list = []
    max_score_possible_list = []
    for i in range(len(num_only_array[:])): #isolating lists that are just picks, not the score of a game played
        win_loss_list = []
        for j in range(len(num_only_array[1])):
            current_pick = num_only_array[i, j]
            if current_pick == 1 or current_pick == -1:
                win_loss_list.append(current_pick)
            else:
                continue

        win_loss_list_week_before = win_loss_list[:-1] #remove the last pick to get the week before confidence score

        #getting absolute confidence and max confidence score possible for that week 
        confidence_score_list = []
        max_value_list = []
        num_of_picks = len(win_loss_list_week_before)
        for i in range(len(win_loss_list_week_before)):
            weekly_score = i / num_of_picks * win_loss_list_week_before[i] #the win/loss get weighted by how many picks came before
            max_value = i / num_of_picks
            confidence_score_list.append(weekly_score)
            max_value_list.append(max_value)
        confidence_list.append(sum(confidence_score_list))#list gets summed to ge the final confidence score
        max_score_possible_list.append(sum(max_value_list))
        print(confidence_list)
    print(confidence_list)
    return(confidence_list, max_score_possible_list)

def confidence_score_2week_before(num_only_array):
    confidence_list = []
    max_score_possible_list = []
    for i in range(len(num_only_array[:])): #isolating lists that are just picks, not the score of a game played
        win_loss_list = []
        for j in range(len(num_only_array[1])):
            current_pick = num_only_array[i, j]
            if current_pick == 1 or current_pick == -1:
                win_loss_list.append(current_pick)
            else:
                continue

        win_loss_list_week_before = win_loss_list[:-2] #remove the last pick to get the week before confidence score

        #getting absolute confidence and max confidence score possible for that week 
        confidence_score_list = []
        max_value_list = []
        num_of_picks = len(win_loss_list_week_before)
        for i in range(len(win_loss_list_week_before)):
            weekly_score = i / num_of_picks * win_loss_list_week_before[i] #the win/loss get weighted by how many picks came before
            max_value = i / num_of_picks
            confidence_score_list.append(weekly_score)
            max_value_list.append(max_value)
        confidence_list.append(sum(confidence_score_list))#list gets summed to ge the final confidence score
        max_score_possible_list.append(sum(max_value_list))
        print(confidence_list)
    print(confidence_list)
    return(confidence_list, max_score_possible_list)


def basic_plot(data_array, num_only_array, confidence_list, max_possible_list):
    #generating actual win/loss from original data (not manually)
    team_week = data_array[:, 0]
    win_loss_list = []
    for i in range(len(num_only_array[:])): 
        for j in range(len(num_only_array[1])):
            current_pick = num_only_array[i, j]
            if current_pick == 1 or current_pick == -1:
                continue
            if re.search('[0-9]+-[0-9]+ [a-zA-Z]\S*', current_pick):
                split_pick = current_pick.split() #gets the last item with is score and W/L
                print(split_pick)
                win_loss_list.append(split_pick[1]) #takes either Win of Loss and addsd to list 
                break

    #creating the dataframe to graph
    data = {'team_week': team_week,
            'confidence_score': confidence_list,
            'max_possible': max_possible_list,
            'bills_win_or_loss': win_loss_list}
    df = pd.DataFrame(data, columns = ['team_week', 'confidence_score', 'max_possible', 'bills_win_or_loss'])
    df['reverse_max'] = df['max_possible'] * -1

    #generating the graph of normalized confidence scores
    palette = {'Win':"blue",
                'Loss':"red"}
    sns.barplot( x = "team_week", y = "confidence_score", data = df, hue = "bills_win_or_loss", palette = palette, dodge=False)
    plt.plot(df["team_week"], df["max_possible"], color = 'black')
    plt.plot(df["team_week"], df["reverse_max"], color = 'black')
    plt.xticks(rotation = 90, fontsize = 10)
    plt.xlabel("Week/Opponent", fontsize = 10)
    plt.ylabel("Confidence Interval", fontsize = 10)
    plt.legend(title = "Bill's Win or Loss")
    plt.tight_layout()
    plt.show()

def scatter_confidence_plot(num_only_array):
    #creating a confidence score and max possible score for each pick
    confidence_list = []
    week_list = []
    team_list = []
    pick_number = []
    for i in range(len(num_only_array[:])): #isolating lists that are just picks, not the score of a game played
        team_week = data_array[i, 0]
        win_loss_list = []
        for j in range(len(num_only_array[1])):
            current_pick = num_only_array[i, j]
            if current_pick == 1 or current_pick == -1:
                win_loss_list.append(current_pick)
            else:
                break
        #getting absolute confidence and max confidence score possible for that week 
        num_of_picks = len(win_loss_list)
        for j in range(len(win_loss_list)):
            weekly_score = (j + 1) / num_of_picks * win_loss_list[j] #the win/loss get weighted by how many picks came before
            confidence_list.append(weekly_score)
            week_list.append(i + 1)
            team_list.append(team_week)
            pick_number.append(j + 1)


    data = {'team_week': team_list,
            'confidence_score': confidence_list,
            'pick_number': pick_number}
    df = pd.DataFrame(data, columns = ['team_week', 'confidence_score', 'pick_number'])
    sns.scatterplot(x ='team_week', y = 'confidence_score', data = df, size = 'pick_number', hue = 'pick_number', sizes=(20, 400), legend = False)
    plt.legend([],[], frameon=False)
    plt.xticks(rotation = 90, fontsize = 12)
    plt.xlabel("Week/Opponent", fontsize = 12)
    plt.ylabel("Confidence Scores", fontsize = 12)
    plt.tight_layout()
    plt.show()



############################################################################

df = make_df('C:/Users/msind/Box Sync/code/github_projects/hptb/HPTB_2021.csv')
data_array = make_array(df)
num_only_array = win_loss_convert(data_array)
confidence_list, max_score_list = confidence_score(num_only_array)
basic_plot(data_array, num_only_array, confidence_list, max_score_list)
scatter_confidence_plot(num_only_array)
confidence_list_week_before, max_score_list_week_before = confidence_score_week_before(num_only_array)

confidence_list_2week_before, max_score_list_2week_before = confidence_score_2week_before(num_only_array)



print(data_array)
print(num_only_array)
df2 = pd.DataFrame()
df2['confidence_scores'] =confidence_list_2week_before 
print(df2)

df2['confidence_scores'].to_clipboard(excel=True, sep=None, index=False, header=None)