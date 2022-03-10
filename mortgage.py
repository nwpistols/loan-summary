import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter


def _p(k, P, N, alpha):
    '''
    Outstanding principal at step k
    '''
    A = P * (np.power(1 + alpha, k))
    A -= P * np.power(1 + alpha, N) * (np.power(1 + alpha, k) - 1) / (np.power(1 + alpha, N) - 1)
    return A


def _I(k, P, N, alpha):
    '''
    Interest payment at step k
    '''
    return alpha * _p(k, P, N, alpha)
 
    
def _f(P, N, alpha):
    '''
    Fixed monthly payment
    '''
    return alpha * np.power(1 + alpha, N) * P / (np.power(1 + alpha, N) - 1)


def _X(k, P, N, alpha):
    '''
    Amount payed toward principle at step k
    '''
    return _f(P, N, alpha) - alpha * _p(k, P, N, alpha)


def _total(P, N, alpha):
    return N * _f(P, N, alpha)


def _style(df):
    return (
        df
        .style
        .format("${:,.0f}", subset=pd.IndexSlice['Sale Price':,:])
        .format("{:,.3f}", subset=pd.IndexSlice['Interest to Principal Ratio',:])
        .format("{:.1%}", subset=pd.IndexSlice['PMI Rate',:])
        .applymap(lambda x: "font-weight:bold")
        .set_table_styles(
            [
                dict(selector="tbody", props=[('border', 'solid thin #ddd')]),
                dict(selector="thead", props=[('border', 'solid thin #ddd')]),
            ]

        )
    )


class Mortgage:

    def __init__(self, sale_price, num_years, interest_rate, down_payment_fraction, PMI_rate=0.01, tax_rate=1.4e-2):
        self.sale_price = sale_price
        self.num_years = num_years
        self.num_months = self.num_years * 12
        self.interest_rate = interest_rate
        self.down_payment_fraction = down_payment_fraction
        self.down_payment = self.down_payment_fraction * self.sale_price
        
        # Annual PMI rate
        self.PMI_rate = PMI_rate
        
        self.tax_rate = tax_rate
        self.principal = self.sale_price - self.down_payment
        self.summary_table = None
        self.compute_summary()
        
    def summary(self):
        if not self.summary_table:
            self.compute_summary()
        return self.summary_table
    
    def _compute_total_PMI_payment(self):
        '''Compute the total Mortgage insurance payed.
        This is the sum of monthly PMI payments until the outstanding 
        principal is 80% of principal'''
        alpha = self.interest_rate / 12.
        P = self.principal
        N = self.num_months
        ns = np.arange(N)
        ys = ns / 12.

        # Principal balance
        ps = _p(ns, P, N, alpha)
        return np.sum([ self.monthly_PMI_payment for p in ps if p > 0.8 * self.principal])
    
    def compute_summary(self):
        r = self.interest_rate
        alpha = r / 12.
        N = self.num_months
        P = self.principal

        if self.down_payment_fraction < 0.2:
            self.monthly_PMI_payment = self.PMI_rate * self.principal / 12.
        else:
            self.monthly_PMI_payment = 0    
        self.total_PMI_payment = self._compute_total_PMI_payment()
			
        total_tax = self.tax_rate * self.sale_price * self.num_years
        monthly_tax = self.tax_rate * self.sale_price / 12.
        total_loan_repayment = _total(P, N, alpha)
        grand_total = total_loan_repayment + total_tax + self.down_payment + self.total_PMI_payment
        
        total_interest = total_loan_repayment - self.principal
        
        self.df_summary = pd.DataFrame(
            [
                ("Down Payment Fraction", self.down_payment_fraction),
				("Interest Rate", self.interest_rate),
				("PMI Rate", self.PMI_rate),

                ("Sale Price", self.sale_price),
                ("Down Payment", self.down_payment),
                ("Principal loan", self.principal),

                ("Interest to Principal Ratio", np.round(total_interest / self.principal, 2)),
                ("Monthly PMI Insurance", self.monthly_PMI_payment),
                ("Monthly Principal + Interest", _f(P, N, alpha)),
                ("Monthly loan Payment", _f(P, N, alpha) + self.monthly_PMI_payment),
                ("Monthly Payment + Tax (approx)", _f(P, N, alpha) + self.monthly_PMI_payment + monthly_tax),
                
                ("Total Tax Paid*", total_tax),
                ("Total loan Payment", total_loan_repayment),
                ("Total Interest Paid", total_interest),
                ("Total PMI Paid*", self.total_PMI_payment),
                ("Total Cost (no tax)", total_loan_repayment + self.down_payment + self.total_PMI_payment),
                ("Grand Total", grand_total),
                
            ],
            columns=["item", 'value']
        ).set_index('item')
        self.df_summary.index.name = None
        self.summary_table = _style(self.df_summary)
        
    def plot(self):
        alpha = self.interest_rate / 12.
        P = self.principal
        N = self.num_months
        ns = np.arange(N)
        ys = ns / 12.

        # Principal balance
        ps = _p(ns, P, N, alpha)

        # Principal payments
        xs = _X(ns, P, N, alpha)

        # Interest payments
        Is = _I(ns, P, N, alpha)

        frac_price = 0.8
        
        # 80% of the total price
        price_q80 = frac_price * self.sale_price
        
        plt.figure(figsize=(15, 4), dpi=100)
        with sns.axes_style('whitegrid'):
            plt.plot(ys, ps, '-.', label="Principal Balance") 
            plt.plot(ys, price_q80 * np.ones_like(ys), '-r', label="{:,.0%} of sale price".format(frac_price))
            plt.legend(loc=3)
            plt.xticks(np.arange(N / 12 + 1))
            plt.xlim(0, N / 12)
            plt.ticklabel_format(style="sci")
            plt.title('Principal Balance As A Function Of Time', fontweight="bold")
            plt.xlabel('Time (year)')
				
            plt.gca().yaxis.set_major_formatter(
                FuncFormatter(
                    lambda x, pos: "${:,.0f}".format(x)
            ))

        plt.figure(figsize=(15, 4), dpi=100)
        with sns.axes_style('whitegrid'):
            plt.title('Monthly Principal And Interest Payments As A Function Of Time', fontweight="bold")
            plt.plot(ys, Is, '-.', label="Interest Payment") 
            plt.plot(ys, xs, '-.', label="Principal Payment") 
            plt.legend(loc=3)
            plt.xlabel('Time (year)')
            plt.xticks(np.arange(N / 12 + 1))
            plt.xlim(0, N / 12)
            plt.gca().yaxis.set_major_formatter(
                FuncFormatter(
                    lambda x, pos: "${:,.0f}".format(x)
            ))
            
    def get_num_insurance_payments(self):
        alpha = self.interest_rate / 12.
        P = self.principal
        N = self.num_months
        ns = np.arange(N)
        ys = ns / 12.

        # Principal balance
        ps = p(ns, P, N, alpha)

        frac_price = 0.8
        
        # 80% of the total price
        price_q80 = frac_price * self.sale_price
        
        if max(ps) >= price_q80:
            n_months = 0
            return n_months
        else:
            pass


def table_multiple_interest_rates(sale_price, down_payment_percent, interest_rates=[], num_years=30, pmi_rate=0.005, tax_rate=1.4e-2):
    """ Compare multiple mortgages with different interest rates"""
    assert isinstance(interest_rates, list), "interest_rates must be a list"
    assert len(interest_rates) > 0
    
    list_mortgages = [
        Mortgage(
            sale_price=sale_price,
            num_years=num_years,
            interest_rate=interest_rate,
            down_payment_fraction=down_payment_percent / 100.,
            PMI_rate=pmi_rate,
            tax_rate=tax_rate
        ) for interest_rate in interest_rates
    ]

    df = pd.concat(
        [m.df_summary for m in list_mortgages],
        axis=1
    )
    
    df.columns = ["{:.2%}".format(m.interest_rate) for m in list_mortgages]
    df.columns.name = "Interest Rate"
    table = _style(df).background_gradient(cmap="Reds", axis=1)
    return table, list_mortgages


def table_multiple_down_payment_percents(sale_price, interest_rate, down_payment_percents=[], num_years=30, pmi_rate=0.005, tax_rate=1.4e-2):
    """compare multiple mortgages with different down payment percentages"""
    list_mortgages = [
        Mortgage(
            sale_price=sale_price,
            num_years=num_years,
            interest_rate=interest_rate,
            down_payment_fraction=pct / 100.,
            PMI_rate=pmi_rate,
            tax_rate=tax_rate
        ) for pct in down_payment_percents
    ]

    df = pd.concat(
        [m.df_summary for m in list_mortgages],
        axis=1
    )
    
    df.columns = ["{:.1%}".format(m.down_payment_fraction) for m in list_mortgages]
    df.columns.name = "Down Payment Rate"
    table = _style(df).background_gradient(cmap="Reds", axis=1)
    return table, list_mortgages

