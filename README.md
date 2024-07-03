## The traveling David problem
I'm going on a long road trip and I'd like to get an idea for how many miles I'll be driving. This code estimates solutions to the traveling salesman problem using various approaches, and incorporates driving distance estimates and uncertainty.

## Usage
1. Install requirements: `pip install -r requirements.txt`
2. Run: `python driving.py`

## Example output:
```
Nearest Neighbor: 10043 mi                      ['NOVA', 'NYC', 'Bost', 'Main', 'Nash', 'Denv', 'Salt', 'Phoe', 'Los ', 'Seat', 'Rich', 'Rich']
MST Approximation: 10043 mi                     ['NOVA', 'NYC', 'Bost', 'Main', 'Nash', 'Denv', 'Salt', 'Phoe', 'Los ', 'Seat', 'Rich', 'Rich']
Simulated Annealing: 9583 mi                    ['NOVA', 'Rich', 'Nash', 'Denv', 'Phoe', 'Los ', 'Salt', 'Seat', 'Main', 'Main', 'Bost', 'NYC']
Genetic Algorithm #1: 9277 mi                   ['NOVA', 'Rich', 'Bost', 'NYC', 'Main', 'Seat', 'Seat', 'Salt', 'Los ', 'Phoe', 'Denv', 'Nash']
Genetic Algorithm #2: 9298 mi                   ['NOVA', 'Rich', 'Bost', 'NYC', 'Main', 'Salt', 'Salt', 'Seat', 'Los ', 'Phoe', 'Denv', 'Nash']
Genetic Algorithm #3: 9438 mi                   ['NOVA', 'Rich', 'NYC', 'Bost', 'Main', 'Seat', 'Salt', 'Phoe', 'Los ', 'Salt', 'Denv', 'Nash']
Genetic Algorithm #4: 9699 mi                   ['NOVA', 'Rich', 'NYC', 'Bost', 'Main', 'Seat', 'Seat', 'Los ', 'Phoe', 'Salt', 'Denv', 'Nash']
Genetic Algorithm #5: 9699 mi                   ['NOVA', 'Rich', 'NYC', 'Bost', 'Main', 'Seat', 'Seat', 'Los ', 'Phoe', 'Salt', 'Denv', 'Nash']
Monte Carlo Sim #1: 10032 mi                    ['NOVA', 'Denv', 'Seat', 'Salt', 'Los ', 'Phoe', 'Nash', 'Rich', 'Rich', 'Bost', 'Main', 'NYC']
Monte Carlo Sim #2: 10173 mi                    ['NOVA', 'Rich', 'Main', 'Bost', 'NYC', 'Nash', 'Denv', 'Phoe', 'Los ', 'Los ', 'Salt', 'Seat']
Monte Carlo Sim #3: 10446 mi                    ['NOVA', 'Main', 'Bost', 'NYC', 'Seat', 'Los ', 'Phoe', 'Denv', 'Denv', 'Salt', 'Nash', 'Rich']
Monte Carlo Sim #4: 10488 mi                    ['NOVA', 'Rich', 'Nash', 'Phoe', 'Seat', 'Seat', 'Los ', 'Salt', 'Denv', 'NYC', 'Bost', 'Main']
Monte Carlo Sim #5: 10676 mi                    ['NOVA', 'Nash', 'Denv', 'Salt', 'Los ', 'Los ', 'Seat', 'Phoe', 'Rich', 'Main', 'Bost', 'NYC']

Simulated Annealing with Uncertainty
Mean tour length: 6187.53 miles
Standard deviation: 942.30 miles
95% Confidence Interval: (5999.62, 6375.45) miles
Best Tour: 5441 mi                              ['NOVA', 'NYC', 'Bost', 'Main', 'Seat', 'Seat', 'Salt', 'Los ', 'Phoe', 'Denv', 'Nash', 'Rich']
```