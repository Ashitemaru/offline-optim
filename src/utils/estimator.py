# proxy side function, can be implemented in the client SDK
class TimestampExtrapolator:
    def __init__(
        self, first_frame_sts, first_frame_recv_ts, mode=0
    ):  # 0 for naive mode, 1 for Kalman filter
        self.first_frame_sts = first_frame_sts
        self.first_frame_recv_ts = first_frame_recv_ts
        self.mode = mode

        if self.mode == 1:
            self.w = [1, 0]
            self.p = [[1, 0], [0, 1e5]]

    def update(self, frame_sts, frame_recv_ts):
        if self.mode == 0:
            pass
        elif self.mode == 1:
            recv_time_diff = frame_recv_ts - self.first_frame_recv_ts
            residual = (
                frame_sts
                - self.first_frame_sts
                - recv_time_diff * self.w[0]
                - self.w[1]
            )

            k = [0, 0]
            k[0] = self.p[0][0] * recv_time_diff + self.p[0][1]
            k[1] = self.p[1][0] * recv_time_diff + self.p[1][1]

            kLambda = 1
            tpt = kLambda + recv_time_diff * k[0] + k[1]
            k[0] /= tpt
            k[1] /= tpt

            self.w[0] = self.w[0] + k[0] * residual
            self.w[1] = self.w[1] + k[1] * residual

            p00 = (
                1
                / kLambda
                * (
                    self.p[0][0]
                    - (k[0] * recv_time_diff * self.p[0][0] + k[0] * self.p[1][0])
                )
            )
            p01 = (
                1
                / kLambda
                * (
                    self.p[0][1]
                    - (k[0] * recv_time_diff * self.p[0][1] + k[0] * self.p[1][1])
                )
            )
            self.p[1][0] = (
                1
                / kLambda
                * (
                    self.p[1][0]
                    - (k[1] * recv_time_diff * self.p[0][0] + k[1] * self.p[1][0])
                )
            )
            self.p[1][1] = (
                1
                / kLambda
                * (
                    self.p[1][1]
                    - (k[1] * recv_time_diff * self.p[0][1] + k[1] * self.p[1][1])
                )
            )
            self.p[0][0] = p00
            self.p[0][1] = p01
        else:
            raise NotImplementedError

    def reset(self, first_frame_sts, first_frame_recv_ts):
        if self.mode == 0:
            pass
        elif self.mode == 1:
            self.w = [1, 0]
            self.first_frame_sts = first_frame_sts
            self.first_frame_recv_ts = first_frame_recv_ts
        else:
            raise NotImplementedError

    def extrapolate_local_time(self, frame_sts):
        if self.mode == 0:
            return self.first_frame_recv_ts + (frame_sts - self.first_frame_sts)
        elif self.mode == 1:
            sts_diff = frame_sts - self.first_frame_sts
            return self.first_frame_recv_ts + (sts_diff - self.w[1]) / self.w[0]
        else:
            raise NotImplementedError


# proxy side function, can be implemented in the client SDK
class ClientProcTimeEstimator:
    def __init__(
        self, avg_dec_time, avg_render_time, avg_proc_time, mode=0
    ):  # 0 for EWMA, 1 for Kalman filter
        self.avg_dec_time = avg_dec_time
        self.avg_render_time = avg_render_time
        self.avg_proc_time = avg_proc_time

        self.mode = mode

        if self.mode == 0:
            self.ewma_alpha = 0.05
        elif self.mode == 1:
            self.w = [1, 0]
            self.p = [[1, 0], [0, 1e5]]

    def update(self, dec_time, render_time, proc_time):
        if self.mode == 0:
            self.avg_dec_time = (
                self.ewma_alpha * dec_time + (1 - self.ewma_alpha) * self.avg_dec_time
            )
            self.avg_render_time = (
                self.ewma_alpha * render_time
                + (1 - self.ewma_alpha) * self.avg_render_time
            )
            self.avg_proc_time = (
                self.ewma_alpha * proc_time + (1 - self.ewma_alpha) * self.avg_proc_time
            )

        elif self.mode == 1:
            proc_time_diff = proc_time - self.avg_proc_time
            residual = -proc_time_diff * self.w[0] - self.w[1]

            k = [0, 0]
            k[0] = self.p[0][0] * proc_time_diff + self.p[0][1]
            k[1] = self.p[1][0] * proc_time_diff + self.p[1][1]

            kLambda = 1
            tpt = kLambda + proc_time_diff * k[0] + k[1]
            k[0] /= tpt
            k[1] /= tpt

            self.w[0] = self.w[0] + k[0] * residual
            self.w[1] = self.w[1] + k[1] * residual

            p00 = (
                1
                / kLambda
                * (
                    self.p[0][0]
                    - (k[0] * proc_time_diff * self.p[0][0] + k[0] * self.p[1][0])
                )
            )
            p01 = (
                1
                / kLambda
                * (
                    self.p[0][1]
                    - (k[0] * proc_time_diff * self.p[0][1] + k[0] * self.p[1][1])
                )
            )
            self.p[1][0] = (
                1
                / kLambda
                * (
                    self.p[1][0]
                    - (k[1] * proc_time_diff * self.p[0][0] + k[1] * self.p[1][0])
                )
            )
            self.p[1][1] = (
                1
                / kLambda
                * (
                    self.p[1][1]
                    - (k[1] * proc_time_diff * self.p[0][1] + k[1] * self.p[1][1])
                )
            )
            self.p[0][0] = p00
            self.p[0][1] = p01
        else:
            raise NotImplementedError

    def get_proc_time_estimate(self):
        if self.mode == 0:
            return self.avg_proc_time
        elif self.mode == 1:
            return self.avg_proc_time - self.w[1] / self.w[0]
