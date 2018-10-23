"""
10/4/18 burnsca@amazon.com
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import json



class XGBoost():
    obj_master_list = {'reg-linear', 'reg-logistic', 'binary-logistic', 'binary-logitraw', 'binary-hinge', \
                       'gpu-reg-linear', 'gpu-reg-logistic', 'gpu-binary-logistic', 'gpu-binary-logitraw',
                       'count-poisson', 'survival-cox', \
                       'multi-softmax', 'multi-softprob', 'rank-pairwise', 'rank-ndcg', 'rank-map', 'reg-gamma',
                       'reg-tweedie'}

    def __init__(self, event):
        self.__event = event
        self.__verified = False
        self.__msg = ""
        self.__hyperparams = {}

    @property
    def hyperparams(self):
        return self.__hyperparams

    @property
    def verified(self):
        return self.__verified

    @property
    def msg(self):
        return self.__msg

    def verify(self):
        """
        verify that event passed to the Lambda function has the correct values for the XGBoost algo
            :param event:
            :return: boolean
        """

        if 'hyperparams' not in self.__event:
            self.__msg = "'hyperparams' value missing from API call."
        else:
            self.__hyperparams = self.__event['hyperparams']
            # XGBoost requires an Objective
            if 'objective' not in self.__hyperparams:
                self.__msg = "'hyperparams' value provided must missing the 'objective' key"
            else:
                obj = self.__hyperparams['objective']
                if obj in XGBoost.obj_master_list:
                    self.__verified = True
                else:
                    self.msg = "'hyperparams{'objective'} provided but objective value is not valid. Must be one of the " \
                           "following: https://github.com/dmlc/xgboost/blob/master/doc/parameter.rst#learning-task-parameters"
