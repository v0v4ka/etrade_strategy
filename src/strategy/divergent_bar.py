import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover, TrailingStrategy



class DivergentBar(TrailingStrategy):
    def init(self):
        super().init()
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
        self.cancel_price = None

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
        ao_down = ao < ao.shift(1)
        ao_up = ao > ao.shift(1)
        bullish = upper_half & ao_down & local_min & (open_ < close) & no_cross
        bearish = lower_half & ao_up & local_max & (open_ > close) & no_cross
        result = pd.Series(0, index=close.index)
        result[bullish] = 1
        result[bearish] = -1
        # Store condition matrix for analysis (used by LLM context builder)
        try:
            self._cond_matrix = pd.DataFrame({
                'upper_half': upper_half.astype(int),
                'lower_half': lower_half.astype(int),
                'local_min': local_min.astype(int),
                'local_max': local_max.astype(int),
                'ao_down': ao_down.astype(int),
                'ao_up': ao_up.astype(int),
                'no_cross': no_cross.astype(int),
                'signal': result
            })
        except Exception:
            pass
        return result.values

    def place_divergent_order(self, direction):
        """Place new divergent order (limit + SL) for current bar.
        direction: 1 (bullish) or -1 (bearish)."""
        high = self.data.High[-1]
        low = self.data.Low[-1]
        if direction > 0:
            stop = high
            sl = low - (high - low)
            self.buy(stop=stop, sl=sl)
            self.cancel_price = low
        elif direction < 0:
            stop = low
            sl = high + (high - low)
            self.sell(stop=stop, sl=sl)
            self.cancel_price = high

    def update_profit_trailing_sl(self, n: int = 5, buffer: float = 0.0):
        """
        Move stop-loss only when the structural level is already in PROFIT zone.
        Long:
            structural = max(low over last n bars) - buffer
            Apply only if structural > entry_price (locks profit)
        Short:
            structural = min(high over last n bars) + buffer
            Apply only if structural < entry_price
        Stops only tighten (never loosen).
        """
        if self.position.size == 0:
            return
        window = min(len(self.data.Close), n)
        lows = self.data.Low[-window:]
        highs = self.data.High[-window:]
        struct_long_sl = lows.max() - buffer
        struct_short_sl = highs.min() + buffer

        for trade in self.trades:
            if trade.is_long:
                # Require structural level in profit (above entry)
                if struct_long_sl > trade.entry_price:
                    if trade.sl is None or struct_long_sl > trade.sl:
                        trade.sl = struct_long_sl
            elif trade.is_short:
                # Require structural level in profit (below entry)
                if struct_short_sl < trade.entry_price:
                    if trade.sl is None or struct_short_sl < trade.sl:
                        trade.sl = struct_short_sl


    def next(self):
        # --- Main per-bar logic ---
        super().next()
        signal = self.divergent[-1]

        # Manage existing position (reversal or trail)
        if self.position.size != 0:
            opposite = (signal > 0 and self.position.is_short) or (signal < 0 and self.position.is_long)
            if signal != 0 and opposite:
                # Reverse immediately
                self.position.close()
                self.place_divergent_order(signal)
            else:
                # Trail only if profitable
                if self.position.pl > 0:
                    self.update_profit_trailing_sl(n=3)
            return

        # Cancel pending orders if current bar invalidates prior setup relative to cancel_price
        # (restored logic from previous version)
        if self.cancel_price is not None and self.orders:
            for order in list(self.orders):
                if order.is_long and self.cancel_price >= self.data.Low[-1]:
                    order.cancel()
                elif order.is_short and self.cancel_price <= self.data.High[-1]:
                    order.cancel()


        # Flat: handle new signal
        if signal == 0:
            return
        # Cancel any stale pending orders before placing fresh one
        if self.orders:
            for o in list(self.orders):
                o.cancel()

        self.place_divergent_order(signal)