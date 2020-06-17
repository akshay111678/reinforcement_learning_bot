import numpy as np
import math
from alpha_vantage.timeseries import TimeSeries
from nsetools import Nse
from agent.agent import Agent
from state.state import State
from keras.models import load_model

# prints formatted price
def formatPrice(n):
    return ("" if n < 0 else "") + "{0:.2f}".format(abs(n))


# returns the sigmoid
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def get_data(stock):
    ts = TimeSeries(key='4ETMGNJDLMPGKZZU', output_format='pandas')
    # nse = Nse()
    stock1_data, meta_data = ts.get_daily(symbol='NSE:' + stock, outputsize='full')
    stock1_data = stock1_data[stock1_data.index > '2007-12-31']
    stock1_data['Date'] = stock1_data.index
    stock1_data = stock1_data.reset_index(drop=True)
    stock1_data['Open'] = stock1_data['1. open']
    stock1_data['High'] = stock1_data['2. high']
    stock1_data['Low'] = stock1_data['3. low']
    stock1_data['Close'] = stock1_data['4. close']
    stock1_data['Volume'] = stock1_data['5. volume']
    return stock1_data
def getStockVolVec(stock_name_data):
    vol = []
    for line in stock_name_data['Volume']:
        vol.append(line)
    return vol
def plot_price_chart(stock1_data,stock1):
    x1 = np.array(stock1_data['Date'])
    y1 = stock1_data['Close']
    y12 = stock1_data['Volume']
    plt.title("" + stock1 + " Stock Performance Over years")
    plt.xlabel("Year")
    plt.ylabel("Price in rupees")
    plt.plot(x1, y1)
    ax2 = plt.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('volume', color=color)  # we already handled the x-label with ax1
    ax2.plot(x1, y12, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.show()
def split_train_model(split,data):
    training = len(data) - split
    test = split
    # Training Data
    pd_data1_train = data[0:training]
    # Test Data
    pd_data1_test = data[training:training + test]
    pd_data1_test.reset_index(drop='True',inplace=True)
    # volume of stock
    vol1_train = getStockVolVec(data)
    return pd_data1_train,pd_data1_test,vol1_train,test,training
def find_benchmark(training, data_train, start_balance):
    # Benchmark Model
    # Initialize state and set benchmarking model
    total_Prof = []
    done = False
    Act_datasize = training
    # Benchmark Model
    data1_train = data_train['Open']
    data1_train.reset_index(drop='index')
    data1_date = data_train['Date']
    data1_date.reset_index(drop='index')
    Act_Bench_Stock1_Bal = int(np.floor((start_balance / 2) / data1_train[0]))
    Act_Bench_Open_cash = start_balance / 2

    ### Program to calculate benchmark profit
    # sell 10% of stock in 10 intervals
    interval = int(Act_datasize / 10)
    Total_Stock1_Amount = 0
    stocks1Value = 0
    Act_stocks1 = np.floor(Act_Bench_Stock1_Bal / 10)
    # print(str(Act_stocks1))
    remaining_stock1 = Act_Bench_Stock1_Bal
    ttl = 0
    Benchmark_Port_Value = []

    for j in range(interval, Act_datasize + 1, interval):
        # print("closing prices : " + str(data1_train[j-1]) )
        Price_closing_Stock1 = data1_train[j - 1]
        date_stock1 = data1_date[j - 1].strftime('%Y-%m-%d')
        # print(date_stock1)
        stocks1Value = Act_stocks1 * Price_closing_Stock1
        remaining_stock1 = remaining_stock1 - Act_stocks1
        # print("J is:"+ str(j))
        Stock1_Port_value = remaining_stock1 * Price_closing_Stock1
        Act_Bench_Open_cash = Act_Bench_Open_cash + stocks1Value  # + stocks2Value  # Adding 10% sold value into open cash

        Total_Portfolio_value = Act_Bench_Open_cash + Stock1_Port_value  # + Stock2_Port_value
        Benchmark_Port_Value.append([date_stock1, Total_Portfolio_value])

    # print ("total_Test_Benchmark_amount : " +  str(Total_Portfolio_value))
    Training_Benchmark_Portfolio_Value = Total_Portfolio_value
    return Total_Portfolio_value,remaining_stock1,Benchmark_Port_Value
      # +" and " + stock2 + " stocks:  " + str(remaining_stock2)
def train_model(episode_count,start_balance,data_train,training,date):
    from os import path
    # Define arrays to store per episode values
    total_Prof = []
    total_stock1bal = []
    total_open_cash = []
    total_port_value = []
    total_days_played = []
    batch_size = 64
    # Training run
    for e in range(episode_count + 1):
        print("..........")
        print("Episode " + str(e) + "/" + str(episode_count))

        Bal_stock1 = int(np.floor((start_balance / 2) / data_train[0]))
        open_cash = start_balance / 2

        datasize = training
        done = False
        total_profit = 0
        reward = 0
        max=0

        # Initialize Agent
        agent = Agent(5)
        agent.inventory1 = []
        for i in range(Bal_stock1):
            agent.inventory1.append(data_train[0])
        # Timestep delta to make sure that with time reward increases for taking action
        # timestep_delta=0
        # Running episode over all days in the datasize
        for t in range(datasize):
            # print("..........")
            # print(pd_data1_train.iloc[t,0])
            state_class_obj = State(data_train, Bal_stock1, open_cash, t)
            state_array_obj = state_class_obj.getState()
            action = agent.act(state_array_obj)

            change_percent_stock1 = ( state_class_obj.Stock1Price - state_class_obj.fiveday_stock1) / state_class_obj.fiveday_stock1 * 100
            # profit=data1_train[t]-agent.inventory1(-1)
            # print("change_percent_stock1:  "+str(change_percent_stock1))

            # if action not in [0,1,2]:
            #     reward= reward-1000
            # decide_reward(action,data_train)
            if action == 0:  # buy stock 1
                if state_class_obj.Stock1Price > state_class_obj.open_cash:
                    '''
                    print("Buy stock 1 when it did not have cash, so bankrupt, end of episode")
                    reward=-reward_timedelta*10
                    done = True
                    '''

                    reward = reward-4000
                    # done = True
                    # end episode

                else:
                    # print("In Buy stock 1")
                    agent.inventory1.append(data_train[t])
                    Bal_stock1_t1 = len(agent.inventory1)
                    # Bal_stock2_t1 = len(agent.inventory2)
                    open_cash_t1 = state_class_obj.open_cash - state_class_obj.Stock1Price  # Here we are buying 1 stock

                    # needs to be reviewed

                    if (state_class_obj.open_cash < 500):
                        reward = reward-2000
                    elif (0.1 * Bal_stock1_t1 > Bal_stock1):
                        reward = reward-(1000* Bal_stock1_t1)
                    # elif (abs(change_percent_stock1) <= 2):
                    #     reward = reward-2000
                    else:
                        reward = reward-(change_percent_stock1 * 1000)

            if action == 1:  # sell stock 1
                if state_class_obj.Stock1Blnc < 1:
                    # print("sold stock 2 when it did not have stock 2, so bankrupt, end of episode")
                    reward = reward-4000
                    # done = True
                    # end episode
                else:
                    # print("In sell stock 1")
                    bought_price1 = agent.inventory1.pop(0)
                    Bal_stock1_t1 = len(agent.inventory1)
                    total_profit += data_train[t] - bought_price1
                    # Bal_stock2_t1 = len(agent.inventory2)
                    open_cash_t1 = state_class_obj.open_cash + state_class_obj.Stock1Price  # State[0] is the price of stock 1. Here we are selling 1 stoc

                    if (0.1 * Bal_stock1_t1 > Bal_stock1):
                        reward = reward-(1000 * Bal_stock1_t1)
                    # elif (abs(change_percent_stock1) <= 2):
                    #     reward = -1000
                    elif total_profit>200:
                        reward=reward+(2000 * total_profit)
                    else:
                        reward = reward +(change_percent_stock1 * 100)  # State[0] is the price of stock 1. Here we are selling 1 stock

                    # total_profit += data1_train[t] - bought_price1
                # print("reward for sell stock1 " + str(reward))

            if action == 2:  # Do nothing action
                # if (abs(change_percent_stock1) <= 2):
                #     reward = 100
                if (state_class_obj.open_cash < 0.05 * start_balance):
                    reward += 2000
                else:
                    reward= reward-2000

                Bal_stock1_t1 = len(agent.inventory1)
                # Bal_stock2_t1 = len(agent.inventory2)
                open_cash_t1 = open_cash
            # print("Do nothing")

            if t == datasize - 1:
                # print("t==datasize")
                done = True
                next_state_class_obj = State(data_train, Bal_stock1_t1, open_cash_t1, t)
                next_state_array_obj = next_state_class_obj.getState()
            else:
                next_state_class_obj = State(data_train, Bal_stock1_t1, open_cash_t1, t + 1)
                next_state_array_obj = next_state_class_obj.getState()

            agent.memory.append((state_array_obj, action, reward, next_state_array_obj, done))
            # print("Action is "+str(action)+" reward is" + str(reward))

            Bal_stock1 = Bal_stock1_t1
            # Bal_stock2 = Bal_stock2_t1
            open_cash = open_cash_t1

            if done == True:
                total_Prof.append(total_profit)
                total_stock1bal.append(len(agent.inventory1))
                # total_stock2bal.append(len(agent.inventory2))
                total_open_cash.append(state_class_obj.open_cash)
                total_port_value.append(state_class_obj.portfolio_value)
                total_days_played.append(t)
                print("--------------------------------")
                state_class_obj.reset()
                break

            if len(agent.memory) > batch_size:
                agent.expReplay(batch_size)
        print(reward)
        if reward>max:
            max=reward
            agent.model.save("models/model_"+date+"-max")

        if e % 30 == 0:
            agent.model.save("models/model_"+date+"-" + str(e))
    if path.exists("models/model_"+date+"-max"):
        model_name="model_"+date+"-max"
    else:
        model_name="model_"+date+"-" + str(episode_count)
    return model_name
def test_model(episode_count, data_test,data_test_open, start_balance, model_name):
    # Define arrays to store per episode values
    Act_datasize = len(data_test)
    Act_Bench_Stock1_Bal = int(np.floor((start_balance / 2) / data_test_open[0]))
    Act_Bench_Open_cash = start_balance / 2
    model = load_model("models/" + model_name)
    # Actual run
    episode_count = 0
    # Define arrays to store per episode values
    total_Prof = []
    total_stock1bal = []
    total_open_cash = []
    total_port_value = []
    total_days_played = []
    Act_total_Prof = []
    Act_total_stock1bal = []
    Act_total_open_cash = []
    Act_total_port_value = []
    Act_total_days_played = []
    actions_done_perday = []
    portfolio_value = []
    for e in range(1):  # here we run only for 1 episode, as it is Test run
        Bal_stock1_t2 = Act_Bench_Stock1_Bal
        done = False
        open_cash_t2 = Act_Bench_Open_cash
        total_profit = 0
        reward = 0

        # Initialize Agent
        agent_test = Agent(8, is_eval=True, model_name=model_name)
        # agent = Agent(8)

        agent_test.inventory1 = []
        for i in range(Bal_stock1_t2):
            agent_test.inventory1.append(data_test_open[0])
            # Timestep delta to make sure that with time reward increases for taking action
        timestep_delta = 0

        # Running episode over all days in the datasize
        for t in range(Act_datasize):
            print("..........")

            print(data_test.iloc[t, 0])
            state_class_obj = State(data_test_open, Bal_stock1_t2, open_cash_t2, t)
            state_array_obj = state_class_obj.getState()
            action = agent_test.act(state_array_obj)

            print("Total portfolio value: " + str(state_class_obj.portfolio_value) +
                  "  stock 1 number: " + str(len(agent_test.inventory1))+ "  open cash" + str(state_class_obj.open_cash))

            # reward should be more as time goes further. We will remove reward_timedelta from actual reward
            # reward_timedelta=(datasize-t)*timestep_delta

            change_percent_stock1 = (state_class_obj.Stock1Price - state_class_obj.fiveday_stock1) / state_class_obj.fiveday_stock1 * 100

            # print("change_percent_stock1:  "+str(change_percent_stock1))
            # print("change_percent_stock2:  "+str(change_percent_stock2))
            if action == 0:  # buy stock 1
                if state_class_obj.Stock1Price > state_class_obj.open_cash:
                    '''
                    print("Buy stock 1 when it did not have cash, so bankrupt, end of episode")
                    reward=-reward_timedelta*10
                    done = True
                    '''
                    done = True
                    # end episode

                else:
                    # print("In Buy stock 1")
                    agent_test.inventory1.append(data_test_open[t])
                    Bal_stock1_t2 = len(agent_test.inventory1)
                    open_cash_t2 = state_class_obj.open_cash - state_class_obj.Stock1Price  # Here we are buying 1 stock

            if action == 1:  # sell stock 1
                if state_class_obj.Stock1Blnc < 1:
                    # print("sold stock 2 when it did not have stock 2, so bankrupt, end of episode")

                    done = True
                    # end episode
                else:
                    # print("In sell stock 1")
                    agent_test.inventory1.pop(0)

                    Bal_stock1_t2 = len(agent_test.inventory1)
                    # Bal_stock2_t2 = len(agent_test.inventory2)
                    open_cash_t2 = state_class_obj.open_cash + state_class_obj.Stock1Price  # State[0] is the price of stock 1. Here we are buying 1 stoc

            if action == 2:  # Do nothing action
                Bal_stock1_t2 = len(agent_test.inventory1)
                # Bal_stock2_t2 = len(agent_test.inventory2)
            # print("Do nothing")

            if t == Act_datasize - 1:
                # print("t==datasize")
                done = True
                next_state_class_obj = State(data_test_open, Bal_stock1_t2, open_cash_t2, t)
                next_state_array_obj = next_state_class_obj.getState()
            else:
                # print("t!=datasize"+str(open_cash_t2))
                next_state_class_obj = State(data_test_open, Bal_stock1_t2, open_cash_t2, t + 1)
                next_state_array_obj = next_state_class_obj.getState()

            # print("Action is "+str(action)+" reward is" + str(reward))

            actions_done_perday.append(action)
            portfolio_value.append(next_state_class_obj.portfolio_value)

            if done == True:
                print("--------------------------------")
                print("Total Profit: " + formatPrice(next_state_class_obj.portfolio_value - start_balance))
                print("Total No. of days played: " + str(t) + "  out of overall days:  " + str(Act_datasize))
                print("Total portfolio value: " + str(next_state_class_obj.portfolio_value) +
                      "  stock 1 number: " + str(len(agent_test.inventory1)) + "  open cash" + str(
                    next_state_class_obj.open_cash))
                # + "  stock 2 number: " + str(len(agent_test.inventory2))

                Act_total_Prof.append(total_profit)
                Act_total_stock1bal.append(len(agent_test.inventory1))
                # Act_total_stock2bal.append(len(agent_test.inventory2))
                Act_total_open_cash.append(state_class_obj.open_cash)
                Act_total_port_value.append(state_class_obj.portfolio_value)
                Act_total_days_played.append(t)

                print("--------------------------------")
                state_class_obj.reset()
                break
    opencash = state_class_obj.open_cash

    return total_profit, portfolio_value, opencash, Act_total_days_played

    # Test Stock Prices and actions taken by agent Stock Plot
