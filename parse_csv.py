import os,sys
import pandas as pd


def parse_csv(file:str):
        
    df= pd.read_csv(file)

    games = []
    #to read headers:
    #print(df.columns)

    for game in df["game_moves"]:
        moves = [game[i:i+2] for i in range(0,len(game),2)]
        movePairs = [(ord(m[0])-96,int(m[1])) for m in moves]
        games.append(movePairs)

    #convert from hexadecimal to value
    #each game is generally 120 chars long; 2 chars/move
    #letter, number
    return games
            
