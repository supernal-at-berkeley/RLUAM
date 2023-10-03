import numpy as np



def autoregressive_possion_rate(df):

    """
    Build autoregressive possion schedules

    Parameters:
    schedule (dataframe): Single day input schedule out of demand_profile_generator.py
    """

    df['schedule'] = np.ceil(df['schedule'])
    df['schedule_hr'] = df['schedule'] // 60
    df.loc[df['schedule_hr'] == 24, 'schedule_hr'] = 23
    df_grouped = df.groupby(['schedule_hr', 'od', 'date']).count().reset_index()
    hourly_rate = df_grouped.groupby(['schedule_hr', 'od'])['schedule'].mean().reset_index()

    lax_dtla_rate = hourly_rate[hourly_rate['od'] == 'LAX_DTLA']['schedule'].to_numpy()
    dtla_lax_rate = hourly_rate[hourly_rate['od'] == 'DTLA_LAX']['schedule'].to_numpy()

    return lax_dtla_rate, dtla_lax_rate



def pois_generate(rate, alpha):
    lambda_0_0 = rate[0]
    x_0 = np.random.poisson(lambda_0_0)
    output = [x_0]
    for i in range(1,len(rate)):
        lambda_1_0 = rate[i]
        lambda_1 = lambda_1_0 + (x_0 - lambda_0_0)*lambda_1_0/lambda_0_0 * alpha

        x_0 = np.random.poisson(lambda_1)
        lambda_0_0 = lambda_1_0
        output.append(x_0)

    return output