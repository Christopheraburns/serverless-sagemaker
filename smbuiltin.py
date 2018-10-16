"""
10/4/18 burnsca@amazon.com
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import json
import sagemaker
from sagemaker import get_execution_role


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


    def getcontainer(self, region):
        """
        xgboost specific code goes here to set up the training container
        :param region:
        :return:
        """
        from sagemaker.amazon.amazon_estimator import get_image_uri

        container = get_image_uri(region, 'xgboost')

        return container


    def buildtrainer(self, container, bucket, session):
        """
        build a sagemaker train specific to this built-in algo
        :return:
        """

        # get the ARN of the executing role (to pass to Sagemaker for training)
        role = get_execution_role()

        trainer = sagemaker.estimator.Estimator(container,
                                                role,
                                                train_instance_count=1,
                                                train_instance_type='ml.m4.xlarge',
                                                output_path='s3://{}'.format(bucket),
                                                sagemaker_session=session)

        return trainer


    def sethyperparameters(self, trainer, hyperparams):
        """
        set xgboost specific hyperparameters here
        :param trainer:
        :return:
        """
        trainer.set_hyperparameters(max_depth=5,
                                    eta=0.2,
                                    gamma=4,
                                    min_child_weight=6,
                                    subsample=0.8,
                                    silent=0,
                                    objective=hyperparams['objective'],
                                    eval_metric='rmse',
                                    num_round=100)

        return trainer
