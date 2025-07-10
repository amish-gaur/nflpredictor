
# generaltest.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1) Load & clean the new data
data = pd.read_csv("/Users/amish/Downloads/superbowl 2/general game predictor/games.csv", low_memory=False)
data.columns = data.columns.str.strip()

# 2) Create win flag for each game
data['home_win'] = (data['home_score'] > data['away_score']).astype(int)
data['away_win'] = (data['away_score'] > data['home_score']).astype(int)

# 3) Build per-team aggregated statistics
teams = pd.concat([
    data[['home_team','home_win','home_score','away_score']].rename(
        columns={'home_team':'team','home_win':'win',
                 'home_score':'score_for','away_score':'score_against'}),
    data[['away_team','away_win','away_score','home_score']].rename(
        columns={'away_team':'team','away_win':'win',
                 'away_score':'score_for','home_score':'score_against'})
])

# Calculate team-level metrics
team_stats = teams.groupby('team').agg(
    team_games=('win', 'count'),
    team_wins=('win', 'sum'),
    team_avg_score=('score_for', 'mean'),
    team_avg_against=('score_against', 'mean')
).reset_index()
team_stats['team_win_pct'] = team_stats['team_wins'] / team_stats['team_games']
team_stats['team_net_points'] = team_stats['team_avg_score'] - team_stats['team_avg_against']

# Merge back stats for home and away to build game-level dataset
df = data.merge(
    team_stats[['team','team_win_pct','team_net_points','team_avg_score','team_avg_against']],
    left_on='home_team', right_on='team'
).rename(columns={
    'team_win_pct':'home_win_pct','team_net_points':'home_net_points',
    'team_avg_score':'home_avg_score','team_avg_against':'home_avg_against'
}).drop(columns=['team'])

df = df.merge(
    team_stats[['team','team_win_pct','team_net_points','team_avg_score','team_avg_against']],
    left_on='away_team', right_on='team'
).rename(columns={
    'team_win_pct':'away_win_pct','team_net_points':'away_net_points',
    'team_avg_score':'away_avg_score','team_avg_against':'away_avg_against'
}).drop(columns=['team'])

# 4) Define feature columns and target
feature_cols = [
    'home_win_pct','home_net_points','home_avg_score','home_avg_against',
    'away_win_pct','away_net_points','away_avg_score','away_avg_against'
]
target_col = 'home_win'  # 1 if home team won, else 0

# 5) Drop any rows with missing values
df = df.dropna(subset=feature_cols + [target_col])

# 6) Prepare training data
X = df[feature_cols]
y = df[target_col]

# 7) Train/test split & model training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestClassifier(n_estimators=500, random_state=42)
model.fit(X_train, y_train)
print(f"Model trained. Test accuracy: {model.score(X_test, y_test):.3f}")

# 8) Interactive prediction for arbitrary matchups
def predict_matchup(home, away):
    # Validate teams
    if home not in team_stats['team'].values or away not in team_stats['team'].values:
        print("Team not recognized. Available teams:", ", ".join(team_stats['team']))
        return
    # Fetch stats
    h = team_stats[team_stats['team']==home].iloc[0]
    a = team_stats[team_stats['team']==away].iloc[0]
    row = {
        'home_win_pct': h['team_win_pct'],
        'home_net_points': h['team_net_points'],
        'home_avg_score': h['team_avg_score'],
        'home_avg_against': h['team_avg_against'],
        'away_win_pct': a['team_win_pct'],
        'away_net_points': a['team_net_points'],
        'away_avg_score': a['team_avg_score'],
        'away_avg_against': a['team_avg_against'],
    }
    X_game = pd.DataFrame([row], columns=feature_cols)
    pred = model.predict(X_game)[0]
    winner = home if pred == 1 else away
    print(f"🏈 Prediction: {home} vs {away} -> Predicted Winner: {winner}")

if __name__ == "__main__":
    home_team = input("Enter HOME team (exact name): ").strip()
    away_team = input("Enter AWAY team (exact name): ").strip()
    predict_matchup(home_team, away_team)
