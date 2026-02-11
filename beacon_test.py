from beacon_european import MarketData, EuropeanOption, BlackScholesEngine, OptionType


class Node:
    """A basic dependency node mimicking a Beacon 'G-Sheet' cell."""
    def __init__(self, value=None, func=None, dependencies=None):
        self._value = value
        self._func = func
        self._dependencies = dependencies or []
        self._cache = None

    def get(self):
        if self._func:
            # Recompute if any dependency would normally signal (simplified here)
            return self._func(*[d.get() for d in self._dependencies])
        return self._value

    def set(self, value):
        self._value = value
        # In a real graph, this would trigger an invalidation of downstreams

# --- Defining the Beacon Graph ---

# 1. Market Data Nodes (Inputs)
spot_node = Node(value=100.0)
vol_node = Node(value=0.20)
rate_node = Node(value=0.05)

# 2. MarketData Wrapper (Internal Node)
def make_market(s, v, r):
    return MarketData(spot=s, vol=v, rate=r)

market_node = Node(func=make_market, dependencies=[spot_node, vol_node, rate_node])

# 3. Instrument Node
option_node = Node(value=EuropeanOption(strike=105, expiry=1.0, option_type=OptionType.CALL))

# 4. Pricing Engine Node
engine_node = Node(value=BlackScholesEngine())

# 5. Final Result Node (The "Output Cell")
def calculate_price(opt, mkt, eng):
    opt.set_engine(eng)
    return opt.price(mkt)

price_node = Node(func=calculate_price, dependencies=[option_node, market_node, engine_node])

# --- Usage ---
print(f"Initial Price: {price_node.get():.4f}")

# Update a single leaf node (the Spot)
spot_node.set(110.0)
print(f"Updated Price (Spot=110): {price_node.get():.4f}")