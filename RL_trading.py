from functions import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt



def reinforcement_bot(Stock_name,start_balance,episode_training,plotting=False):
    today=dt.datetime.today().strftime('%d%m%Y')
    stock1=Stock_name
    stock1_data=get_data(stock1)
    if plotting==True:
        plot_price_chart(stock1_data,stock1)
    data_train_,data_test_,vol_train,test,training=split_train_model(1000,stock1_data)
    data_train=data_train_["Open"]
    data_test=data_test_["Open"]
    data_test.reset_index()
    total_port_val,rem_stock,bench_port_val_test=find_benchmark(training,data_train_,start_balance)
    print("Benchmark_Profit is  " + str(total_port_val) + " with " + stock1 + " Stocks:  " + str(rem_stock))
    model_name=train_model(episode_training,start_balance,data_train,training,today)
    #finding benchmark for test
    total_bench_port_val,rem_stock_test,test_bench_port_val=find_benchmark(test,data_test_,start_balance)
    print("Benchmark_Profit is  " + str(total_bench_port_val) + " with " + stock1 + " Stocks:  " + str(rem_stock_test))
    total_profit,portfolio_value,opencash,Act_total_days_played=test_model(episode_training,data_test_,data_test,start_balance,model_name)
    if Act_total_days_played[0] != test - 1:
        diff=test-Act_total_days_played[0]
        port_new=[]
        for i in range(diff):
            port_new.append(portfolio_value[0])
        portfolio_value=port_new.append(portfolio_value)
    #plotting the final graph of benchmark and portfolio
    pd_bm = pd.DataFrame.from_records(test_bench_port_val)
    pd_bm[0] = pd.to_datetime(pd_bm[0], format='%Y/%m/%d')
    x1 = np.array(data_test_['Date'])
    y1 = portfolio_value
    x2 = pd_bm[0]
    y2 = pd_bm[1]
    plt.title("Portfolio Value vs Benchmark Over Test Data")
    plt.xlabel("Days", rotation=90)
    plt.ylabel("Portfolio Value in rupees")
    plt.plot(x1, y1)
    plt.plot_date(x2, y2, c='red', marker='v', linestyle='-')
    plt.xticks(rotation=90)
    plt.plot(x1, y1, '-', color='blue');
    plt.legend(('Trading Model', 'Benchmark'))
    plt.show()


if __name__ == '__main__':
    reinforcement_bot(Stock_name='Reliance',start_balance=10000,episode_training=90)