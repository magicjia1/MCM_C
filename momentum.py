

class Momentum():

        def __init__(self, player_no,sets, games, points, server_advantage, unforced_errors, double_faults,
                     consecutive_points,
                     successful_serves, successful_returns, break_points_successful, ace_situation, net_shots,net_shots_won,
                     consecutive_point_victor_before):

            self.player_no = player_no
            self.sets = sets
            self.games = games
            self.points = points
            self.server_advantage = server_advantage
            self.unforced_errors = unforced_errors
            self.double_faults = double_faults

            self.successful_serves = successful_serves
            self.successful_returns = successful_returns
            self.break_points_successful = break_points_successful
            self.ace_situation = ace_situation
            self.net_shots = net_shots
            self.net_shots_won = net_shots_won

            # 增加连胜场数
            self.consecutive_points = consecutive_points
            self.consecutive_point_victor_before = consecutive_point_victor_before

        def calculate_score(self):
            score = self.sets * 1.856 + self.games * 3.207 +  self.points * 6.136
            # score = self.sets * 1.856 + self.points * 6.136

            if self.player_no == self.server_advantage :
                score += 1 * (8.542)
                # score += 1 * (0)
            else:
                score += 0
            # Handling error situations
            score += self.unforced_errors * (-29.063)
            score += self.double_faults * (-15.458)

            #  #Handling consecutive points
            # if self.player_no == self.consecutive_point_victor_before:
            #     if  self.consecutive_points>=2:
            #       score += (self.consecutive_points-1) * 14.156

            # Handling technical performance
            score += self.successful_serves * 7.929
            score += self.successful_returns * 9.549
            score += self.break_points_successful * 17.493
            # score += self.ace_situation * 16.586

            # Handling net shots
            if  self.net_shots_won:

               score +=  17.69
            else:
                score +=  -5.480

            return score