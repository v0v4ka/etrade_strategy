import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover



class DivergentBar(Strategy):
    def init(self):
        # Precompute AO indicator as bars
        self.ao = self.I(
            self.ao_indicator, self.data.High, self.data.Low
        )
        # Precompute Alligator indicator (returns 3 lines: jaw, teeth, lips) with colors
        self.jaw, self.teeth, self.lips = self.I(
            self.alligator_indicator, self.data.Close,
            name=['Jaw', 'Teeth', 'Lips'],
            color=['blue', 'red', 'green']
        )
        # Precompute Divergent Bar indicator
        self.divergent = self.I(
            self.divergent_bar_indicator,
            self.data.Open, self.data.High, self.data.Low, self.data.Close,
            self.jaw, self.teeth, self.lips, self.ao,
            name='DivergentBar', color='purple'
        )

    def ao_indicator(self, high, low):
        """
        Calculate the Awesome Oscillator (AO) indicator manually.
        AO = SMA(Median Price, 5) - SMA(Median Price, 34)
        Median Price = (High + Low) / 2
        """
        median_price = (high + low) / 2
        sma_5 = pd.Series(median_price).rolling(window=5).mean()
        sma_34 = pd.Series(median_price).rolling(window=34).mean()
        return sma_5 - sma_34

    def alligator_indicator(self, close):
        """
        Calculate the Alligator indicator (Jaw, Teeth, Lips).
        Jaw: 13-period SMA shifted 8 bars
        Teeth: 8-period SMA shifted 5 bars
        Lips: 5-period SMA shifted 3 bars
        Returns three pandas Series for overlaying on the plot.
        """
        jaw = pd.Series(close).rolling(window=13).mean().shift(8)
        teeth = pd.Series(close).rolling(window=8).mean().shift(5)
        lips = pd.Series(close).rolling(window=5).mean().shift(3)
        return jaw, teeth, lips
    
    def divergent_bar_indicator(self, open_, high, low, close, jaw, teeth, lips, ao):
        """
        Vectorized indicator for divergent bars.
        Returns 1 for bullish divergent bar, -1 for bearish, 0 otherwise.
        Bullish:
            - close > (high + low) / 2 (upper half)
            - low < min(jaw, teeth, lips)
            - ao < ao.shift(1)
            - low < min(low.shift(1), low.shift(2), low.shift(3))
        Bearish:
            - close < (high + low) / 2 (lower half)
            - high > max(jaw, teeth, lips)
            - ao > ao.shift(1)
            - high > max(high.shift(1), high.shift(2), high.shift(3))
        """
        jaw = pd.Series(jaw)
        teeth = pd.Series(teeth)
        lips = pd.Series(lips)
        ao = pd.Series(ao)
        low = pd.Series(low)
        high = pd.Series(high)
        close = pd.Series(close)
        min_alligator = pd.concat([jaw, teeth, lips], axis=1).min(axis=1)
        max_alligator = pd.concat([jaw, teeth, lips], axis=1).max(axis=1)
        upper_half = close > (high + low) / 2
        lower_half = close < (high + low) / 2
        local_min = low < pd.concat([low.shift(1), low.shift(2), low.shift(3)], axis=1).min(axis=1)
        local_max = high > pd.concat([high.shift(1), high.shift(2), high.shift(3)], axis=1).max(axis=1)
        # Bar does not cross any alligator line
        no_cross = (
            (high < min_alligator) | (low > max_alligator)
        )
        bullish = upper_half & (low < min_alligator) & (ao < ao.shift(1)) & local_min & (open_ < close) & no_cross
        bearish = lower_half & (high > max_alligator) & (ao > ao.shift(1)) & local_max & (open_ > close) & no_cross
        result = pd.Series(0, index=close.index)
        result[bullish] = 1
        result[bearish] = -1
        return result.values

    def place_divergent_order(self, direction):
        """
        Places a limit order with stop loss for the given direction.
        direction: 1 for bullish (buy), -1 for bearish (sell)
        """
        high = self.data.High[-1]
        low = self.data.Low[-1]
        if direction > 0:
            stop = high
            stop_loss = low - (high - low)
            self.buy(stop=stop, sl=stop_loss)
        elif direction < 0:
            stop = low
            stop_loss = high + (high - low)
            self.sell(stop=stop, sl=stop_loss)

    def next(self):
        # --- Trade logic for single order, every bar, only fully executed orders ---
        
        # don't place new orders if there are open ones
        if self.position:
            return
        
        # Cancel all pending orders if new divergent bar found (regardless of position)
        if self.divergent[-1] != 0:
            for order in self.orders:
                order.cancel()
        
        # Cancel pending orders if stop loss would have been hit by current bar
        for order in self.orders:
            if order.is_long and order.sl is not None and order.sl >= self.data.Low[0]:
                order.cancel()
            elif order.is_short and order.sl is not None and order.sl <= self.data.High[0]:
                order.cancel()
        # If no open position, place new order on divergent bar
        if self.divergent[-1] != 0 and self.position.size == 0:
            self.place_divergent_order(self.divergent[-1])
        # If open position in opposite direction, close and open new order
        if self.divergent[-1] > 0 and self.position.is_short:
            self.position.close()
            self.place_divergent_order(self.divergent[-1])
        elif self.divergent[-1] < 0 and self.position.is_long:
            self.position.close()
            self.place_divergent_order(self.divergent[-1])