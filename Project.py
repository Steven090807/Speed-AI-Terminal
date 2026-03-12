import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from tabulate import tabulate
import numpy as np
import os


ice_cars_df = pd.read_csv("sport_car_dataset.csv", encoding="latin1")
ev_cars_df = pd.read_csv("electric_vehicles_dataset.csv", encoding="latin1")
top10_cars_df = pd.read_csv("top10_fastest_car.csv", encoding="latin1")
tcr_racing_df = pd.read_csv("tcr_racing.csv", encoding="latin1")
charging_range_df = pd.read_csv("charging_ range.csv", encoding="latin1")

#seven spaces
space = "       "
two_space = "  "

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def menu():
    print(f"""
{space}--------------------------------------------------------------
{space}                                                                
{space}    ███████╗██████╗ ███████╗███████╗██████╗     █████╗ ██╗      
{space}    ██╔════╝██╔══██╗██╔════╝██╔════╝██╔══██╗   ██╔══██╗██║      
{space}    ███████╗██████╔╝█████╗  █████╗  ██║  ██║   ███████║██║      
{space}    ╚════██║██╔═══╝ ██╔══╝  ██╔══╝  ██║  ██║   ██╔══██║██║     
{space}    ███████║██║     ███████╗███████╗██████╔╝██╗██║  ██║██║    
{space}    ╚══════╝╚═╝     ╚══════╝╚══════╝╚═════╝ ╚═╝╚═╝  ╚═╝╚═╝      
{space}                                                               
{space}        </> Author: Steven Goh | Interactive ML                
{space}                                                                
{space}  ========================================================  ⠀⠀ ⠀⠀  ⠀                 ⠀ ⢀⣠⠴⠖⠛⠛⠛⠛⠛⠛⠟⠛⠛⠛⠛⠛⠛⠛⠻⠿⠿⠿⠿⠿⢿⣿⠽⠽⠿⢷⣒⠦⢄⣀⣀⣀⣀⣀⣀⡀
{space}        Explore Elite Cars: Top Speed, Price & Range                             ⠀  ⢀⣠⠞⠉⠀⠀⠀⠀⠩⡑⠈⢓⠀⠃⠒⡐⢆⡐⣃⣒⠀⢠⠏⡟⠀⡀⠀⢀⢈⢹⣛⢮⡻⣏⡝⢿⡿⢿⡿⢿⡇⡇
{space}  ========================================================                   ⠀    ⢀⡴⠋⠀⠀⢀⣄⠀⠀⢂⠀⠨⠀⢄⢤⢄⢠⠀⢀⠀⠊⠠⣠⠏⣸⡗⡄⡄⠄⠀⢈⠋⢧⣀⣀⡹⣎⡷⣿⣀⣀⣀⣁⣃
{space}                                                                  ⠀⠂⠄⠀⠀⠀⠀⠀⣈⣀⣀⣒⣤⣴⠾⢯⠦⠴⠤⠤⠴⠥⢤⣥⣬⡯⠡⢓⠄⠤⠄⠓⣁⣈⣁⣉⣘⣞⣓⣟⣧⣔⣅⡈⠹⠖⠚⠛⠳⣍⣀⣀⡤⠤⠝⠛⠛⠛⢷⡀
{space}                                                             ⠀⠐⢀⠀⠀⠀⠀⠀⢠⣀⣭⠤⠶⠛⠋⠉⠙⠋⠁⠀⠀⠀⠀⠀⡉⠉⠃⠌⠉⡉⠷⣬⠀⣀⡨⢤⠴⣞⠚⠛⢉⣉⠀⠀⠈⡇⠙⠋⠀⣀⣀⠤⠴⠒⠋⠋⠋⡇⠁⠁⠀ ⣀⢀ ⡈⢧
{space}[0] Press '0' or 'BACK' to cancel the program and return     ⠀⠂⠀⠀⠀⣠⣶⣾⡿⠋⠇⠀⠀⠁⠀⠀⠀⠀⠀⠠⠂⠀⠀⠂⠘⠈⣤⣀⣬⠾⠓⢟⣉⠤⣿⣿⣢⠵⠛⠉⠙⡈⡳⣍⣡⡷⠖⠉⠉⣥⢂⠀⠀⠐⠀ ⢀⠗⠀⠀ ⣤⡿⢿⣿⣾⣿   ⠀      
{space}                                                              ⠀⠂⣴⢿⡽⠋⢹⠒⢶⠓⠒⠒⡻⠮⠦⡧⠭⠐⠨⡭⡭⠬⠔⡛⠋⠙⣨⡬⣖⠏⣻⣥⡾⡛⠏⣑⣵⣷⣯⣗⣦⡀⠻⡀⢰⣞⡀⣀⠂⡀⡀⣀⡐ ⡀⡀⣸⠄  ⢰⣿⣿⡿⣿⣿⠸
{space}    ==== Turbo Electro Menu ====                             ⠀⢠⡾⣩⡿⣡⣴⣶⣗⣒⣂⣬⣴⣭⣯⣏⣩⡡⣴⣔⠤⠄⠴⠀⣤⡾⣻⡷⡟⣿⣷⠟⣁⣄⣠⣴⣿⣿⣿⡋⠻⣷⣵⡠⢹⣸⠄⠀⠀ ⠁⠂⠐⠀ ⠂⢂⡇⠀ ⠀⢸⣿⣿⣧⣿⡽⠀     
{space}[1] Search Car                 [8] Predict Car Stats        ⠀⢸⣷⣿⣼⡟⠙⣻⠿⠿⡁⠀⢱⢾⠿⠿⠿⠿⠿⢛⡖⠂⣄⡾⠷⠶⢞⠛⠛⣿⠟⠉⠁⠀⣺⣿⣧⣾⣞⣯⠱⣾⣿⣇⠀⢿⠀⠀⠀⠂⠀⠀⠀⠀  ⣀⣼⡀ ⠀ ⣿⣿⣿⣯⣿⡇⢰
{space}[2] Fastest Cars (Top 10)      [9] Longest Range EVs        ⠀⣸⣶⣿⣿⠉⢉⠓⠚⢛⠺⠟⠗⡳⠿⠿⡤⠾⠴⣾⣷⠈⠀⡇⠀⠙⠀⠆⣸⡟⠀⠂⠅⢐⣿⡇⣦⢹⣷⢿⣀⢚⣻⡇⠀⠺⣄⠐⣀⣢⢤⠴⣞⣫⣯⣯⣯⠤⠤⠗⠒⠒⡿⣿⣻⣿⡏⠀⠘
{space}[3] Popular Cars Ranking       [10] Fastest Charging EVs    ⠀⣸⣻⣿⣿⣿⣿⠿⣿⣷⣧⣦⣤⣬⣤⣤⣤⠵⢄⣸⣿⠈⠉⢻⣥⣅⣤⣴⣾⡇⠀⠀⠄⢸⡟⣷⣿⣿⣿⠺⠛⢭⣿⡇⠀⢀⣏⠭⠽⠚⣛⣉⡥⠶⠒⠚⠛⠒⠚⠛⠉⠸⠿⠿⠀⠀⠀
{space}[4] Price Distribution                                     ⠀⠀⢻⠠⢽⣿⡻⠿⣾⡿⣿⡿⣿⣿⣿⣿⣿⣿⡿⡀⣸⣿⠀⠀⠸⠿⡿⠿⠿⠋⡇⢀⡰⣠⣾⡇⡿⣋⣀⣏⣻⡦⣼⣾⠓⠋⣁⣴⠴⠛⠛⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
{space}[5] Affordable Supercars                                    ⠀⠸⠿⣿⣿⣾⣮⣯⣭⣿⣛⣻⣻⣿⠿⠿⠥⠼⢿⠛⣿⣧⣶⣶⣿⣥⣭⣉⡹⠽⠟⠋⠁⣸⡇⢷⢿⡏⢸⡆⢁⣼⠛⠚⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
{space}[6] TCR Racing Leaderboard                                   ⠀⠀⠀⠀⠉⠉⠙⠛⠓⠛⠿⠿⠿⠿⢿⣿⣾⣿⣿⣟⣿⣷⣖⣀⣀⣐⣐⣐⣖⣊⣩⣉⣍⣳⠘⢦⡣⠬⣵⣾⠏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
{space}[7] JDM Lengends (Top 10)                                     ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠁⠁⠉⠏⠩⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠛⠓⠒⠛⠒⠛⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
{space}----------------------------------------------------------
{space}[00] Exit
        """)
    
def sub_menu():
    print(f"""

{space}--------------------------------------------------------------
{space}                                                                
{space}    ███████╗██████╗ ███████╗███████╗██████╗     █████╗ ██╗     
{space}    ██╔════╝██╔══██╗██╔════╝██╔════╝██╔══██╗   ██╔══██╗██║     
{space}    ███████╗██████╔╝█████╗  █████╗  ██║  ██║   ███████║██║  ⠀⠀⠀⠀⠀⠀⠀ ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠟⠛⠻⠿⢿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
{space}    ╚════██║██╔═══╝ ██╔══╝  ██╔══╝  ██║  ██║   ██╔══██║██║  ⠀⠀⠀⠀⠀⠀⠀⠀ ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣇⢚⣁⣀⡀⠀⠀⠀⠉⠉⠙⠛⠻⠿⠿⣄⣄⣄⣄⣄⣄
{space}    ███████║██║     ███████╗███████╗██████╔╝██╗██║  ██║██║  ⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⣴⣶⣦⣤⣤⣀⣀⠀⠀⠀⠀⠉⠉⢛⣄
{space}    ╚══════╝╚═╝     ╚══════╝╚══════╝╚═════╝ ╚═╝╚═╝  ╚═╝╚═╝  ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⣠⣿⣿⡿⠿⠿⠿⠛⠯⠭⣤⢤⠤⠤⠀⢚⣻⣿⡍⢀⣛⣛⣻⠿⣿⣿⡏⢠⣷⣶⣤⣀⠀⠾⣣
{space}                                                            ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠿⠛⠉⠁⠀⢀⣀⣤⠴⠖⠒⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠁⠘⠚⠻⠗⣲⠁⣨⣟⣛⢿⣿⣿⣿⣿
{space}        </> Author: Steven Goh | Interactive ML             ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⠀⠀⠀⠀⣠⣿⠿⣿⣿⣿⡿⠛⠉⠀⠀⠀⠀⠀⠈⠀⠊⠁⠒⠀⠀⠀⠀⠀⢀⣤⡤⠠⠀⠀⠀⠀⠀⡠⠊⣠⣿⣿⣎⣠⡴⢟⢶⣕⠹⣿
{space}                                                            ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⠀⠀⠀⣿⠁⢀⣼⠟⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠀⠀⠀⠀⠀⠀⡠⠊⢁⠤⣹⡿⠟⢍⣦⣷⣾⣿⠿⠧⢻⣿
{space}  ========================================================  ⠀⠀ ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠿⣟⢋⠥⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠂⡠⠈⠀⢰⣇⣇⣁⡴⠾⢿⣿⣿⡟⠁⠀⠀⠃⣿
{space}        Explore Elite Cars: Top Speed, Price & Range        ⠀⠀⠀ ⠀⠀⠀⠀⠀⣠⡿⠟⠋⣩⢬⣿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⡄⠀⠀⠀⠀⠀⠀⡠⡪⠀⣠⣾⣷⡽⡿⡥⠀⠀⠀⣹⡿⠀⠀⠀⠀⢠⣿
{space}  ========================================================  ⠀⠀⠀⠀ ⠀⠀⣠⢟⣭⣮⣶⡿⢵⣿⣿⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⡾⠁⠀⠀⠀⠀⠀⡐⢴⢟⣵⣾⣿⠿⣋⢅⡞⠁⠀⠀⣰⣿⠁⠀⠀⠀⠀⣿
{space}                                                            ⠀⠀⠀⠀⠀ ⡿⡑⣿⣿⢿⠟⢀⣼⣿⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⠋⠀⢀⣀⣤⣆⡈⠀⠀⣸⡿⠟⠫⡰⠟⠛⠈⠀⠀⠀⠈⣽⠏⠀⠀⠀⠀⢸
{space}                                                            ⠀ ⠀⠀⠀⡿⣼⢷⡿⡱⠃⣠⣿⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⣀⣠⣄⣀⣀⣀⣀⣀⣴⢯⡕⢲⣞⡉⢍⡛⠉⠁⡪⠛⠉⠄⠀⠁⠀⠀⠠⢀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣠
{space}[0] Press '0' or 'BACK' Return to Main Menu                 ⠀⠀ ⠀⡿⢱⣿⡜⣼⣧⣾⣿⣿⣷⣤⣀⠀⠀⢀⣠⣤⣶⣿⡿⠿⢛⡛⢹⣁⣿⠤⠓⡂⢁⣠⣷⣿⡿⠋⠀⠀⠀⠀⠀⠀⠀⢀⢄⠤⠐⠁⠀⠀⠀⠀⢀⣤⣠
{space}                                                           ⠀⠀ ⠀⠟⢀⣿⡟⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⢟⣋⣥⣦⣽⣶⣿⡿⠏⣂⣦⣶⣾⠟⠋⠙⠊⠠⣄⣤⣀⡀⠀⢀⡠⠊⡵⠋⠀⠀⠀⠀⣀⣴⢚
{space}    ==== Predict Car Menu ====                             ⠀ ⠀⣿⡦⠸⣿⡿⣿⡿⣿⣿⣿⣿⣿⣿⣿⣿⣯⣵⣾⣿⣿⣿⣿⢿⢫⣡⣴⣾⣿⣿⡟⠁⠀⠀⠀⠀⠀⠸⣿⣿⣿⣷⢖⠅⠀⠀⠀⢀⣠⣶⢚
{space}                                                           ⠀⠀ ⡟⠀⠀⠙⢿⣮⣗⡤⢿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠛⠋⠉⠀⠀⠀⢨⣿⣿⣿⠏⠀⠀⠀⠀⠀⠀⠀⠀⣿⡿⠋⠀⠀⠀⣀⣤⣾⢚
{space}[1] Range Prediction by Battery Size                       ⠀ ⠀⡇⠀⠀⠀⠀⠙⠻⢿⣿⣾⣿⣿⡿⠟⠋⢁⢀⡀⠀⠤⠤⠤⠀⣢⡿⠉⣻⡏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠁⠀⠀⣀⣴⢚
{space}[2] Charging Time Prediction                                ⠀ |⣿⡄⠀⠀⠀⠀⠀⠀⠈⠉⠛⠛⠷⠤⣒⣂⠄⠀⠀⠀⠀⣠⣾⠟⠁⢀⠟⠈⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⣴⢚
{space}[3] EV Market Growth Prediction                             ⠀⠀ ⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠿⠟⠛⠛⠉⠀⠀⠀⠀⠀⢀⠀⠀⠀⠀⠀⢀⣠⢚⢚
{space}                                                            ⠀⠀⠀ ⠀⠻⣷⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⠀⠀⠀⠀⠀⠀⠊⠀⠀⣀⣤⣶⢚
{space}                                                            ⠀⠀⠀⠀ ⠀⠀⠀⠻⣷⣶⣤⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣴⢚
{space}----------------------------------------------------------  ⠀⠀⠀⠀⠀ ⠀⠀⠀⠀⠀⠀⠀⠻⣿⣷⣶⣶⣦⣤⣤⣤⣤⣤⣤⣴⣶⢚
{space}[00] Exit
        """)

choice_Auto = ""
sub_choice = ""
car_type_01 = ""
car_name_01 = ""
year_filter = ""
input_count = 0
show_tip = True
menu()

while True:

    if choice_Auto.strip() == "":
        choice = input(f"{space}Enter your choice: ").strip()
        if choice.lower() in ["clear", "delete", "refresh"]:
            if sub_choice == "1":
                clear_screen()
                sub_menu()
            else:
                clear_screen()
                menu()
            continue
    else:
        choice = choice_Auto
        choice_Auto = ""


    if choice.lower() in ["1", "one"]:
        if sub_choice == "1":
            print(f"\n{space}==== OPTION ONE ====")
            battery = float(input(f"{space}Enter battery size (kWh): "))
            if battery == "0":
                sub_choice = ""
                clear_screen()
                menu()
            predicted_range = battery * 7.8
            print(f"{space}Predicted Range: ~{predicted_range:.0f} km\n")
            continue

        if car_type_01 == "":
            print(f"\n{space}==== Search Car ====")
            car_type_01 = input(f"{space}Search for Sport or EV car?: ").strip().lower()
            

        if car_type_01.startswith(("s", "i")):
            search_df = ice_cars_df.copy()
            input_count += 1
            type = "ICE"
        elif car_type_01.startswith("e"):
            search_df = ev_cars_df.copy()
            input_count += 1
            type = "EV"
        elif car_type_01 == "0":
            choice_Auto = ""
            car_type_01 = ""
            clear_screen()
            menu()
            continue
        else:
            print(f"{space}⚠️{two_space}Invalid choice. Please enter Sport or EV. ⚠️\n")
            continue

        if car_name_01 == "":
            car_name_01 = input(f"{space}Enter car name: ").strip()
            if car_name_01 == "0":
                clear_screen()
                menu()
                continue
        if year_filter != "":
            year_filter = ""
        else:
            year_filter = input(f"{space}Enter year (press Enter to skip): ").strip()
            if year_filter == "0":
                clear_screen()
                menu()
                continue

        search_df["Full Name"] = search_df["Car Make"].astype(str) + " " + search_df["Car Model"].astype(str)
        results = search_df[search_df["Full Name"].str.contains(car_name_01, case=False, na=False)]

        if year_filter.isdigit() and "Year" in results.columns:
            results = results[results["Year"] == int(year_filter)]

        if type == "ICE":
            display_cols = [col for col in [
                "Car Make", "Car Model", "Year", "Engine Size (L)", "Horsepower", "Torque (lb-ft)", 
                "MPH Time (seconds)", "Price (in MYR)", "country"
            ] if col in results.columns]
        elif type == "EV":
            display_cols = [col for col in [
                "Car Make", "Car Model", "Year", "Battery_Type", 
                "Batt_Capacity", "Charging_Type", "Price_USD", "Country"
            ] if col in results.columns]

        if not results.empty:
            print(f"\n{space}Search Results:")
            table_str = tabulate(results[display_cols], headers="keys", tablefmt="psql", showindex=False)
            indented_table = "\n".join(space + line for line in table_str.splitlines())
            print(indented_table)
            while True:
                if input_count > 1 and show_tip:
                    print(f"\n{space}[Tip] Type 'Clear' to wipe inputs and return to menu.")
                    repeat_tip = input(f"{space}Need this reminder again? (on/off): ").strip().lower()
                    if repeat_tip == "off":
                        show_tip = False

                back = input(f"\n{space}Continue searching car info?: ").lower().strip()
                if back in ["no", "exit", "quit", "0"]:
                    repeat_tip = -1
                    show_tip = True
                    clear_screen()
                    menu()
                    break  
                elif back in ["yes", "continue", "keep on", "go on", "proceed"]:
                    choice_Auto = "1"
                    car_type_01 = ""
                    car_name_01 = ""
                    year_filter = ""
                    break  
                else:
                    print(f"{space}⚠️{two_space}Please type 'Yes' to continue or 'No'/'0' to return to menu.⚠️\n")

        else:
            print(f"{space}No cars found.\n")
        car_type_01 = ""
        car_name_01 = ""
        year_filter = ""


    elif choice.lower() in ["2", "two"]:
        if sub_choice.strip() == "1":
            print(f"\n{space}==== OPTION TWO ====")
            try:
                battery = float(input(f"{space}Enter battery size (kWh): "))
                if battery == "0":
                    sub_choice = ""
                    clear_screen()
                    menu()
                    continue
                charger = float(input(f"{space}Enter charger power (kW): "))
                if charger == "0":
                    sub_choice = ""
                    clear_screen()
                    menu()
                    continue
                time_hours = battery / charger * 1.6
                time_minutes = time_hours * 60
                hours = int(time_minutes // 60)
                minutes = int(time_minutes % 60)
                print(f"{space}Estimated Charging Time: {hours} hr {minutes} min ({time_hours:.1f} hours)\n")
                continue
            except:
                print(f"{space}⚠️ Invalid input. Please enter numeric values only.\n")
                continue

        year = input(f"{space}Enter year (2023 to 2025): ").strip()

        if year == "0":
            clear_screen()
            menu()
            continue
        if not year.isdigit():
            print(f"{space}Invalid year.")
            continue

        filtered_df = top10_cars_df[top10_cars_df['Year'] == int(year)].copy()
        if filtered_df.empty:
            print(f"{space}No cars found for {year}.")
            continue

        filtered_df['Horsepower'] = (
            filtered_df['Horsepower'].str.replace(r"[^\d]", "", regex=True).astype(int)
        )
        top10_hp = filtered_df.sort_values(by='Horsepower', ascending=False).head(10)

        plt.figure(figsize=(12, 6))
        plt.barh(top10_hp['Car Model'], top10_hp['Horsepower'], color='firebrick')

        for i, (hp, model) in enumerate(zip(top10_hp["Horsepower"], top10_hp["Car Model"])):
            plt.text(hp + 5, i, str(hp), ha="left", va="center", color="black", fontsize=9)

        plt.xlabel("Horsepower (hp)")
        plt.title(f"Top 10 Fastest Cars by Horsepower in {year}")
        plt.gca().invert_yaxis()
        plt.show()



    elif choice.lower() in ["3", "three"]:
        if sub_choice == "1":
            print(f"\n{space}==== OPTION TREE ====")
            ev_growth = ev_cars_df.groupby("Year")["Car Model"].count().reset_index()

            a = np.polyfit(ev_growth["Year"], ev_growth["Car Model"], 2)
            b = np.poly1d(a)

            future_years = np.arange(ev_growth["Year"].min(), 2030)
            predictions = b(future_years)

            plt.figure(figsize=(8, 5))
            plt.plot(ev_growth["Year"], ev_growth["Car Model"], "bo-", label="Actual Data")
            plt.plot(future_years, predictions, "r--", label="Trend Prediction")

            plt.title("EV Market Growth Prediction")
            plt.xlabel("Year")
            plt.ylabel("Number of EV Models")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            sub_choice = ""
            continue


        print(f"\n{space}==== Popular Cars Ranking ====")
        country = input(f"{space}View Malaysia or Global popularity ranking?: ").strip().lower()

        if country in ["my", "malaysia"]:
            country_type = "MY"
            if "popular_MY" not in ice_cars_df.columns:
                print(f"{space}⚠️{two_space}The dataset has no 'popular_MY' column.⚠️")
                continue

            ice_cars_df["popular_MY"] = pd.to_numeric(ice_cars_df["popular_MY"], errors="coerce")

            ranked = ice_cars_df.dropna(subset=["popular_MY"]).sort_values(
                by="popular_MY", ascending=True
            )
        elif country in ["global"]:
            country_type = "GLOB"
            if "popular_global" not in ice_cars_df.columns:
                print(f"{space}⚠️{two_space}The dataset has no 'popular_global' column.⚠️")
                continue

            ice_cars_df["popular_global"] = pd.to_numeric(ice_cars_df["popular_global"], errors="coerce")

            ranked = ice_cars_df.dropna(subset=["popular_global"]).sort_values(
                by="popular_global", ascending=True                
            )
        else:
            clear_screen()
            sub_menu()

        if ranked.empty:
            print(f"{space}⚠️{two_space}No popularity data found in 'popular_global' column.⚠️")
            continue

        top10 = ranked.head(10)

        if country_type == "GLOB":
            display_cols = [col for col in [
                "Car Make", "Car Model", "Year", "popular_global"
            ] if col in top10.columns]
        elif country_type == "MY":
            display_cols = [col for col in [
                "Car Make", "Car Model", "Year", "popular_MY"
            ] if col in top10.columns]

        table_str = tabulate(top10[display_cols], headers="keys", tablefmt="psql", showindex=False)
        indented_table = "\n".join(space + line for line in table_str.splitlines())
        print(indented_table)

        while True:
            back = input(f"\n{space}[0] Return to main menu: ").lower().strip()
            if back in ["exit", "quit", "0"]:
                clear_screen()
                menu()
                break   
            else:
                print(f"{space}⚠️{two_space}Please type '0' to return to menu.⚠️\n")
                continue



    elif choice.lower() in ["4", "four"]:
        car_type_04 = input(f"{space}Search for Sport (ICE) or EV car?: ").strip().lower()

        while True:
            if car_type_04.startswith(("s", "i")):
                search_df = ice_cars_df.copy()
                type = "ICE"
                break
            elif car_type_04.startswith("e"):
                search_df = ev_cars_df.copy()
                type = "EV"
                break
            elif car_type_04 == "0":
                clear_screen()
                menu()
                break
            else:
                print(f"{space}Invalid choice. Please enter Sport or EV.")
                car_type_04 = input(f"{space}Search for Sport (ICE) or EV car?: ").strip().lower()
                continue

        car_name_04 = input(f"{space}Enter car name: ").strip()
        if car_name_04 == "0":
            clear_screen()
            menu()
            continue

        search_df["Full Name"] = search_df["Car Make"].astype(str) + " " + search_df["Car Model"].astype(str)
        results = search_df[search_df["Full Name"].str.contains(car_name_04, case=False, na=False)]

        if type == "ICE":
            display_cols = [col for col in [
                "Car Make", "Car Model", "Year", "Price_MYR"
            ] if col in results.columns]
        elif type == "EV":
            display_cols = [col for col in [
                "Car Make", "Car Model", "Year", "Price_USD"
            ] if col in results.columns]

        if not results.empty:
            print(f"\n{space} ==== Price Distribution ====")
            table_str = tabulate(results[display_cols], headers="keys", tablefmt="psql", showindex=False)
            indented_table = "\n".join(space + line for line in table_str.splitlines())
            print(indented_table)
            print(f"\n{space}[Tip] Typ '1' to look for a specific model.\n")
            while True:
                back = input(f"{space}Continue searching prices?: ").lower().strip()
                if back in ["no", "0"]:
                    clear_screen()
                    menu()
                    break
                elif back in ["yes"]:
                    choice_Auto = "4"
                    break   
                elif back in ["1"]:
                    choice_Auto = "1"
                    car_type_01 = car_type_04
                    car_name_01 = car_name_04
                    year_filter = "1"
                    break   
                else:
                    print(f"{space}⚠️{two_space}Please type 'Yes' to continue or 'No'/'0' to return to menu.⚠️\n")
        else:
            print(f"{space}No cars found.")


    elif choice.lower() in ["5", "five"]:
        print(f"\n{space}==== Filter Your Sport Car ====")
        hp_input = input(f"{space}Your expected minimum horsepower: ").strip()
        budget_input = input(f"{space}Your maximum budget (MYR): ").strip()

        try:
            min_hp = int(hp_input)
            max_budget = int(budget_input)
        except:
            print(f"{space}Invalid input. Please enter numbers only.")
            continue

        ice_cars_df["Horsepower"] = pd.to_numeric(ice_cars_df["Horsepower"], errors="coerce")

        ice_cars_df["Price_MYR"] = (
            ice_cars_df["Price_MYR"].astype(str).str.replace(",", "").str.strip()
        )
        ice_cars_df["Price_MYR"] = pd.to_numeric(ice_cars_df["Price_MYR"], errors="coerce")

        filtered = ice_cars_df[
            (ice_cars_df["Horsepower"] >= min_hp) &
            (ice_cars_df["Price_MYR"] <= max_budget)
        ]

        display_cols = [col for col in [
            "Car Make", "Car Model", "Year", "Horsepower",
            "MPH Time (seconds)", "Price_MYR"
        ] if col in ice_cars_df.columns]

        if not filtered.empty:
            print(f"\n{space}Affordable Sport Cars matching your criteria:")
            filtered_top20 = filtered.head(20)
            table_filter = tabulate(filtered_top20[display_cols], headers="keys", tablefmt="psql", showindex=False)
            indented_table = "\n".join(space + line for line in table_filter.splitlines())
            print(indented_table)

            while True:
                back = input(f"{space}Enter [0] back to main menu: ").strip()
                if back == "0":
                    clear_screen()
                    menu()
                    break
                else:
                    print(f"{space}⚠️{two_space}Please enter [0] to return.⚠️\n")

        else:
            print(f"{space}No cars found matching your criteria.")


    elif choice.lower() in ["6", "six"]:
        print(f"\n{space}==== TCR Racing Leaderboard ====")
        year_input = input(f"{space}Enter year of view (2020-2025): ").strip()
        if not year_input.isdigit():
            print(f"{space}⚠️{two_space}Invalid year input.⚠️")
            continue

        year_input = int(year_input)
        year_data = tcr_racing_df[tcr_racing_df["Year"] == year_input]

        if year_data.empty:
            print(f"{space}⚠️{two_space}No TCR data found for {year_input}.⚠️")
            continue

        location = year_data["Location"].iloc[0]
        top10 = year_data.sort_values("Rnaking").head(10)

        display_cols = ["Drive", "Car Model", "Rnaking", "Race_time", "Location"]
        table_str = tabulate(top10[display_cols], headers="keys", tablefmt="psql", showindex=False)
        indented_table = "\n".join(space + line for line in table_str.splitlines())
        print(indented_table)

        plt.figure(figsize=(10, 6))
        colors = plt.cm.plasma(np.linspace(0, 1, len(top10)))
        plt.barh(top10["Drive"], top10["Rnaking"], color=colors)
 

        for i, (rank, model) in enumerate(zip(top10["Rnaking"], top10["Car Model"])):
            plt.text(rank + 0.1, i, model, ha="left", va="center", color="black", fontsize=9)

        plt.gca().invert_yaxis()

        plt.title(f"TCR {year_input} {location} Championship - Top 10", fontsize=14)
        plt.xlabel("Ranking")
        plt.ylabel("Driver")
        #plt.tight_layout()
        plt.show()



    elif choice.lower() in ["7", "seven"]:
        print(f"\n{space}==== TOP 10 POPULAR JDM CARS ====")

        if "JDM" not in ice_cars_df.columns:
            print(f"{space}⚠️{two_space}The dataset has no 'JDM' column.⚠️")
            continue
        ice_cars_df["JDM"] = pd.to_numeric(ice_cars_df["JDM"], errors="coerce")
        JDM_df = ice_cars_df[ice_cars_df["JDM"].between(1, 10)].copy()

        if JDM_df.empty:
            print(f"{space}⚠️{two_space}No JDM cars found in dataset.⚠️")
            continue
        top10_JDM = JDM_df.sort_values(by="JDM", ascending=True)

        display_cols = [col for col in [
            "JDM", "Car Make", "Car Model", "Year", "Engine Size (L)", 
            "Horsepower", "Torque (lb-ft)", "MPH Time (seconds)", "Price_MYR"
        ] if col in top10_JDM.columns]

        from tabulate import tabulate
        table_str = tabulate(top10_JDM[display_cols], headers="keys", tablefmt="psql", showindex=False)
        indented_table = "\n".join(space + line for line in table_str.splitlines())
        print(indented_table)

        plt.figure(figsize=(10, 5))
        colors = plt.cm.inferno(np.linspace(0, 1, len(top10_JDM)))
        plt.barh(top10_JDM["Car Model"], top10_JDM["JDM"], color=colors)
        plt.title("Top 10 Popular JDM Cars (Ranking)")
        plt.xlabel("Popularity Rank (1 = Most Popular)")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()


    elif choice.lower() in ["8", "eight"]:
        sub_choice = "1"
        clear_screen()
        sub_menu()



    elif choice.lower() in ["9", "nine"]:
        charging_range_df = charging_range_df.sort_values("Range", ascending=False)
        
        plt.figure(figsize=(10, 5))
        plt.plot(charging_range_df["Car_Model"], charging_range_df["Range"], marker='s', linestyle='-', color='g')
        plt.title("Longest Range EVs")
        plt.xlabel("EV Model")
        plt.ylabel("Range (km)")
        plt.xticks(rotation=45)
        plt.gca().invert_yaxis() 
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    elif choice.lower() in ["10", "ten"]:
        charging_range_df = charging_range_df.sort_values("Charging_Speed", ascending=False)

        plt.figure(figsize=(10, 5))
        plt.plot(charging_range_df["Car_Model"], charging_range_df["Charging_Speed"], marker='o', linestyle='-', color='b')
        plt.title("Fastest Charging EVs")
        plt.xlabel("EV Model")
        plt.ylabel("Charging Speed (kW)")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.gca().invert_yaxis() 
        plt.tight_layout()
        plt.show()


    elif choice == "0":
        if sub_choice == "1":
            sub_choice = ""
            clear_screen()
            menu()
        else:
            clear_screen()
            menu()


    elif choice == "00":
        print(f"{space}Exiting program...")
        break

    else:
        print(f"{space}Invalid choice, try again.")
