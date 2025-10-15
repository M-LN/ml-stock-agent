"""""""""

ML Mentor Engine - Simple Version for Deployment

Provides basic model analysis without LLM dependencyML Mentor Engine - Intelligent Model AnalysisML Mentor Engine - Phase 2 Implementation

"""

Analyzes trained ML models and provides actionable recommendationsAnalyserer ML modeller og giver intelligente anbefalinger via LLM

import json

import os""""""

from typing import Dict, List, Any, Optional





def analyze_saved_model(model_id: str, api_key: Optional[str] = None, model: str = "gpt-4o-mini") -> Dict[str, Any]:import jsonimport json

    """

    Analyze a saved model and provide recommendationsimport osimport os

    

    Args:from datetime import datetimefrom datetime import datetime

        model_id: ID of the saved model

        api_key: OpenAI API key (not used in simple version)from typing import Dict, List, Any, Optionalfrom typing import Dict, List, Optional

        model: LLM model (not used in simple version)

    import numpy as npimport numpy as np

    Returns:

        Analysis results with health score and recommendationsfrom openai import OpenAI

    """

    def analyze_saved_model(model_id: str, api_key: Optional[str] = None, model: str = "gpt-4o-mini") -> Dict[str, Any]:from colorama import init, Fore

    # Load training log

    log_file = os.path.join("logs", f"{model_id}_training.json")    """

    

    if not os.path.exists(log_file):    Analyze a saved model and provide recommendationsinit(autoreset=True)

        raise FileNotFoundError(f"Training log not found: {log_file}")

        

    with open(log_file, 'r') as f:

        log_data = json.load(f)    Args:class MLMentorEngine:

    

    # Extract metrics        model_id: ID of the saved model (e.g., "lstm_AAPL_20251015_170914")    """

    metrics = log_data.get('final_metrics', log_data.get('metrics', {}))

    data_info = log_data.get('data_stats', log_data.get('data_info', {}))        api_key: OpenAI API key (optional, uses rule-based if not provided)    ML Mentor Engine der analyserer trænede modeller og giver LLM-baserede anbefalinger.

    

    train_mae = metrics.get('train_mae', 0)        model: LLM model to use (gpt-4o-mini, gpt-4o, gpt-4)    """

    val_mae = metrics.get('val_mae', 0)

            

    # Calculate generalization gap

    if train_mae > 0:    Returns:    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):

        gen_gap = ((val_mae - train_mae) / train_mae) * 100

    else:        Dict with analysis results, health score, and recommendations        """

        gen_gap = 0

        """        Initialiser ML Mentor Engine

    # Build analysis

    analysis = {            

        'metrics': {

            'train_mae': train_mae,    # Load training log        Args:

            'val_mae': val_mae,

            'generalization_gap': gen_gap    log_file = os.path.join("logs", f"{model_id}_training.json")            api_key: OpenAI API key (bruger OPENAI_API_KEY env var hvis None)

        },

        'data_info': {                model: OpenAI model ('gpt-4o-mini' eller 'gpt-4' eller 'gpt-4o')

            'training_samples': data_info.get('training_samples', 0),

            'validation_samples': data_info.get('validation_samples', 0),    if not os.path.exists(log_file):        """

            'total_samples': data_info.get('total_samples', 0)

        },        raise FileNotFoundError(f"Training log not found: {log_file}")        self.api_key = api_key or os.getenv('OPENAI_API_KEY')

        'strengths': [],

        'issues': []            self.model = model

    }

        with open(log_file, 'r') as f:        self.client = None

    # Identify strengths

    if val_mae < 5:        log_data = json.load(f)        

        analysis['strengths'].append({

            'type': 'accuracy',            if self.api_key:

            'description': 'Meget lav validation error - fremragende performance'

        })    # Extract metrics (handle both old and new format)            try:

    if gen_gap < 20:

        analysis['strengths'].append({    metrics = log_data.get('final_metrics', log_data.get('metrics', {}))                self.client = OpenAI(api_key=self.api_key)

            'type': 'generalization',

            'description': 'God generalisering - modellen overfitter ikke'    data_info = log_data.get('data_stats', log_data.get('data_info', {}))                print(f"{Fore.GREEN}✅ LLM initialiseret (OpenAI {model})")

        })

                    except Exception as e:

    # Identify issues

    if val_mae > 15:    train_mae = metrics.get('train_mae', 0)                print(f"{Fore.YELLOW}⚠️ LLM fejl: {str(e)}")

        analysis['issues'].append({

            'severity': 'high',    val_mae = metrics.get('val_mae', 0)                print(f"{Fore.YELLOW}📝 Bruger rule-based fallback")

            'description': 'Høj validation error',

            'impact': 'Modellen giver unøjagtige predictions'            else:

        })

    if gen_gap > 40:    # Calculate generalization gap            print(f"{Fore.YELLOW}ℹ️ Ingen API key - bruger rule-based fallback")

        analysis['issues'].append({

            'severity': 'high',    if train_mae > 0:    

            'description': 'Overfitting detected',

            'impact': 'Modellen generaliser dårligt til ny data'        gen_gap = ((val_mae - train_mae) / train_mae) * 100    def analyze_model(self, model_id: str, training_log: Dict) -> Dict:

        })

    if data_info.get('training_samples', 0) < 300:    else:        """

        analysis['issues'].append({

            'severity': 'medium',        gen_gap = 0        Analyserer en model og giver anbefalinger

            'description': 'Begrænset training data',

            'impact': 'Mere data kunne forbedre performance'            

        })

        # Analyze model performance        Args:

    # Calculate health score

    health_score = calculate_health_score(analysis)    analysis = {            model_id: Model ID (fra training log)

    

    # Generate recommendations        'metrics': {            training_log: Training log data fra JSON fil

    recommendations = generate_recommendations(analysis, log_data)

                'train_mae': train_mae,        

    # Check history

    history = check_model_history(model_id, log_data)            'val_mae': val_mae,        Returns:

    

    return {            'generalization_gap': gen_gap            Dict med analysis, recommendations, og health_score

        'model_id': model_id,

        'health_score': health_score,        },        """

        'analysis': analysis,

        'recommendations': recommendations,        'data_info': {        print(f"\n{Fore.CYAN}🔍 Analyserer model: {model_id}")

        'history': history

    }            'training_samples': data_info.get('training_samples', 0),        



            'validation_samples': data_info.get('validation_samples', 0),        # Step 1: Analyze metrics and detect issues

def calculate_health_score(analysis: Dict[str, Any]) -> int:

    """Calculate model health score (0-100)"""            'total_samples': data_info.get('total_samples', 0)        analysis = self._analyze_metrics(training_log)

    score = 100

            },        

    val_mae = analysis['metrics']['val_mae']

    if val_mae > 20:        'strengths': [],        # Step 2: Get model history (previous versions and their outcomes)

        score -= 40

    elif val_mae > 10:        'issues': []        model_history = self._get_model_history(training_log)

        score -= 20

        }        

    gen_gap = analysis['metrics']['generalization_gap']

    if gen_gap > 50:            # Step 3: Get recommendations (LLM or rule-based) with history context

        score -= 30

    elif gen_gap > 30:    # Identify strengths        if self.client:

        score -= 15

        if val_mae < 5:            recommendations = self._get_llm_recommendations(analysis, training_log, model_history)

    samples = analysis['data_info']['training_samples']

    if samples < 200:        analysis['strengths'].append({        else:

        score -= 20

    elif samples < 400:            'type': 'accuracy',            recommendations = self._get_rule_based_recommendations(analysis)

        score -= 10

                'description': 'Meget lav validation error - modellen performer fremragende'        

    return max(0, min(100, score))

        })        # Step 4: Calculate health score



def generate_recommendations(analysis: Dict[str, Any], log_data: Dict[str, Any]) -> List[Dict[str, Any]]:    elif val_mae < 10:        health_score = self._calculate_health_score(analysis)

    """Generate actionable recommendations"""

    recommendations = []        analysis['strengths'].append({        

    

    for issue in analysis['issues']:            'type': 'accuracy',        return {

        if 'overfitting' in issue['description'].lower():

            recommendations.append({            'description': 'God validation error - modellen er reliable'            'model_id': model_id,

                'priority': 'high',

                'title': 'Reducer Overfitting',        })            'analysis': analysis,

                'why': 'Modellen memorerer training data',

                'action': 'Tilføj regularization eller indsaml mere data',                'recommendations': recommendations,

                'expected': 'Bedre generalisering og 20-30% lavere validation error'

            })    if gen_gap < 15:            'health_score': health_score,

        

        elif 'validation error' in issue['description'].lower():        analysis['strengths'].append({            'history': model_history,

            recommendations.append({

                'priority': 'high',            'type': 'generalization',            'timestamp': datetime.now().isoformat()

                'title': 'Forbedre Accuracy',

                'why': 'Validation error er for høj',            'description': 'Fremragende generalisering - modellen overfitter ikke'        }

                'action': 'Øg model kompleksitet eller træn længere',

                'expected': '30-50% reduktion i validation MAE'        })    

            })

            elif gen_gap < 30:    def _analyze_metrics(self, training_log: Dict) -> Dict:

        elif 'training data' in issue['description'].lower():

            recommendations.append({        analysis['strengths'].append({        """

                'priority': 'medium',

                'title': 'Indsaml Mere Data',            'type': 'generalization',        Analyserer metrics og detector issues

                'why': 'Utilstrækkelig training data',

                'action': 'Udvid data collection period til 2+ år',            'description': 'God generalisering - modellen er stabil'        

                'expected': '15-25% forbedring i performance'

            })        })        Returns:

    

    if not recommendations:                Dict med performance, issues, og strengths

        recommendations.append({

            'priority': 'low',    # Identify issues        """

            'title': 'Optimer Hyperparameters',

            'why': 'Modellen performer godt, men kan fintunes',    if val_mae > 15:        model_type = training_log.get('model_type', 'unknown')

            'action': 'Brug grid search til at finde optimale settings',

            'expected': '5-10% marginal forbedring'        analysis['issues'].append({        final_metrics = training_log.get('final_metrics', {})

        })

                'severity': 'high',        

    return recommendations

            'description': 'Høj validation error',        train_mae = final_metrics.get('train_mae', 0)



def check_model_history(model_id: str, current_log: Dict[str, Any]) -> Dict[str, Any]:            'impact': 'Modellen giver unøjagtige predictions'        val_mae = final_metrics.get('val_mae', 0)

    """Check model version history"""

    symbol = current_log.get('symbol', 'N/A')        })        train_rmse = final_metrics.get('train_rmse', 0)

    model_type = current_log.get('model_type', 'unknown')

    current_version = current_log.get('version', 1)    elif val_mae > 10:        val_rmse = final_metrics.get('val_rmse', 0)

    

    history = {        analysis['issues'].append({        

        'version': current_version,

        'has_parent': current_version > 1,            'severity': 'medium',        # Calculate key indicators

        'related_models': []

    }            'description': 'Moderat validation error',        if train_mae > 0:

    

    # Find related models            'impact': 'Predictions kan have betydelig fejl'            generalization_gap = ((val_mae - train_mae) / train_mae) * 100

    logs_dir = "logs"

    if os.path.exists(logs_dir):        })        else:

        for log_file in os.listdir(logs_dir):

            if log_file.endswith('_training.json'):                generalization_gap = 0

                try:

                    with open(os.path.join(logs_dir, log_file), 'r') as f:    if gen_gap > 50:        

                        other_log = json.load(f)

                            analysis['issues'].append({        # Detect issues

                    if (other_log.get('symbol') == symbol and 

                        other_log.get('model_type') == model_type):            'severity': 'high',        issues = []

                        metrics = other_log.get('final_metrics', other_log.get('metrics', {}))

                        history['related_models'].append({            'description': 'Severe overfitting',        strengths = []

                            'version': other_log.get('version', 1),

                            'val_mae': metrics.get('val_mae', 0),            'impact': 'Modellen memorerer training data i stedet for at lære patterns'        

                            'description': other_log.get('description', '')

                        })        })        # Issue 1: Overfitting

                except:

                    pass    elif gen_gap > 30:        if generalization_gap > 50:

    

    history['related_models'] = sorted(history['related_models'], key=lambda x: x['version'])        analysis['issues'].append({            issues.append({

    

    # Compare with parent            'severity': 'medium',                'type': 'overfitting',

    if history['has_parent']:

        parent = next((m for m in history['related_models'] if m['version'] == current_version - 1), None)            'description': 'Moderate overfitting',                'severity': 'high',

        if parent:

            current_metrics = current_log.get('final_metrics', current_log.get('metrics', {}))            'impact': 'Modellen generaliser ikke optimalt til ny data'                'description': f"Stor generalization gap ({generalization_gap:.1f}%)",

            current_val_mae = current_metrics.get('val_mae', 0)

            parent_val_mae = parent['val_mae']        })                'impact': "Modellen performer dårligt på nye data"

            

            improvement = ((parent_val_mae - current_val_mae) / parent_val_mae * 100) if parent_val_mae > 0 else 0                })

            

            history['outcome'] = {    training_samples = data_info.get('training_samples', 0)        elif generalization_gap > 25:

                'parent_val_mae': parent_val_mae,

                'current_val_mae': current_val_mae,    if training_samples < 200:            issues.append({

                'improvement_percent': improvement,

                'success': improvement > 0,        analysis['issues'].append({                'type': 'overfitting',

                'verdict': 'major_improvement' if improvement > 10 else 

                          'improved' if improvement > 5 else            'severity': 'high',                'severity': 'medium',

                          'minor_improvement' if improvement > 0 else

                          'degraded'            'description': 'Meget lidt training data',                'description': f"Moderat generalization gap ({generalization_gap:.1f}%)",

            }

                        'impact': 'Modellen har ikke nok eksempler at lære fra'                'impact': "Modellen kan være lidt overtilpasset til træningsdata"

            if 'applied_recommendation' in current_log:

                history['applied_recommendation'] = current_log['applied_recommendation']        })            })

    

    return history    elif training_samples < 500:        else:



        analysis['issues'].append({            strengths.append({

class MLMentorEngine:

    """Legacy compatibility class"""            'severity': 'medium',                'type': 'generalization',

    def __init__(self, api_key: Optional[str] = None):

        self.api_key = api_key            'description': 'Begrænset training data',                'description': f"God generalization (gap: {generalization_gap:.1f}%)"

    

    def analyze_model(self, model_id: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:            'impact': 'Mere data kunne forbedre modellens performance'            })

        return analyze_saved_model(model_id, self.api_key, model)

        })        

            # Issue 2: High error rates

    # Calculate health score        if val_mae > 15:

    health_score = calculate_health_score(analysis)            issues.append({

                    'type': 'high_error',

    # Generate recommendations                'severity': 'high',

    recommendations = generate_recommendations(analysis, log_data, api_key, model)                'description': f"Høj validation MAE (${val_mae:.2f})",

                    'impact': "Predictions er upræcise"

    # Check for model history            })

    history = check_model_history(model_id, log_data)        elif val_mae > 10:

                issues.append({

    return {                'type': 'high_error',

        'model_id': model_id,                'severity': 'medium',

        'health_score': health_score,                'description': f"Moderat validation MAE (${val_mae:.2f})",

        'analysis': analysis,                'impact': "Predictions kunne være mere præcise"

        'recommendations': recommendations,            })

        'history': history        else:

    }            strengths.append({

                'type': 'accuracy',

                'description': f"God prediction accuracy (MAE: ${val_mae:.2f})"

def calculate_health_score(analysis: Dict[str, Any]) -> int:            })

    """Calculate overall model health score (0-100)"""        

            # Issue 3: Training curve analysis (for models with epochs)

    score = 100        epochs_data = training_log.get('epochs_data', [])

            if epochs_data and len(epochs_data) > 5:

    # Penalize for validation error            # Check if loss is still decreasing

    val_mae = analysis['metrics']['val_mae']            last_5_losses = [e['loss'] for e in epochs_data[-5:]]

    if val_mae > 20:            first_loss = last_5_losses[0]

        score -= 40            last_loss = last_5_losses[-1]

    elif val_mae > 15:            improvement = ((first_loss - last_loss) / first_loss) * 100

        score -= 30            

    elif val_mae > 10:            if improvement < 1:

        score -= 20                issues.append({

    elif val_mae > 5:                    'type': 'plateau',

        score -= 10                    'severity': 'medium',

                        'description': "Training loss har plateau'd",

    # Penalize for overfitting                    'impact': "Flere epochs vil ikke forbedre modellen"

    gen_gap = analysis['metrics']['generalization_gap']                })

    if gen_gap > 50:            else:

        score -= 30                strengths.append({

    elif gen_gap > 30:                    'type': 'learning',

        score -= 20                    'description': "Modellen lærer stadig ved sidste epoch"

    elif gen_gap > 15:                })

        score -= 10        

            # Issue 4: Data size

    # Penalize for insufficient data        data_stats = training_log.get('data_stats', {})

    samples = analysis['data_info']['training_samples']        training_samples = data_stats.get('training_samples', 0)

    if samples < 200:        

        score -= 20        if training_samples < 100:

    elif samples < 500:            issues.append({

        score -= 10                'type': 'insufficient_data',

                    'severity': 'high',

    return max(0, min(100, score))                'description': f"For få træningsamples ({training_samples})",

                'impact': "Modellen kan ikke lære robuste mønstre"

            })

def generate_recommendations(analysis: Dict[str, Any], log_data: Dict[str, Any],         elif training_samples < 200:

                            api_key: Optional[str] = None, model: str = "gpt-4o-mini") -> List[Dict[str, Any]]:            issues.append({

    """Generate actionable recommendations"""                'type': 'insufficient_data',

                    'severity': 'medium',

    recommendations = []                'description': f"Moderat antal træningsamples ({training_samples})",

                    'impact': "Mere data kunne forbedre performance"

    # Check issues and create recommendations            })

    for issue in analysis['issues']:        else:

        if 'overfitting' in issue['description'].lower():            strengths.append({

            recommendations.append({                'type': 'data_size',

                'priority': 'high',                'description': f"Tilstrækkelig data ({training_samples} samples)"

                'title': 'Reducer Overfitting',            })

                'why': 'Din model memorerer training data i stedet for at lære generelle patterns',        

                'action': 'Tilføj regularization (dropout for LSTM, max_depth for tree models) eller indsaml mere data',        return {

                'expected': 'Generalization gap reduceres med 20-40%, bedre performance på ny data'            'model_type': model_type,

            })            'metrics': {

                        'train_mae': train_mae,

        elif 'validation error' in issue['description'].lower():                'val_mae': val_mae,

            if issue['severity'] == 'high':                'train_rmse': train_rmse,

                recommendations.append({                'val_rmse': val_rmse,

                    'priority': 'high',                'generalization_gap': generalization_gap

                    'title': 'Forbedre Model Accuracy',            },

                    'why': 'Validation error er for høj til reliable predictions',            'issues': issues,

                    'action': 'Øg model kompleksitet (flere layers/neurons for LSTM, flere trees for RF), eller træn længere',            'strengths': strengths,

                    'expected': 'Validation MAE reduceres med 30-50%'            'data_info': {

                })                'training_samples': training_samples,

                        'validation_samples': data_stats.get('validation_samples', 0),

        elif 'training data' in issue['description'].lower():                'total_samples': training_samples + data_stats.get('validation_samples', 0)

            recommendations.append({            }

                'priority': 'high',        }

                'title': 'Indsaml Mere Data',    

                'why': 'Modellen har ikke nok eksempler at lære fra',    def _get_model_history(self, training_log: Dict) -> Dict:

                'action': 'Udvid training perioden til 2+ år ved at ændre data collection period',        """

                'expected': 'Bedre generalisering og 15-30% lavere validation error'        Henter historik for denne model (tidligere versioner og deres resultater)

            })        

            Returns:

    # If no major issues, suggest optimization            Dict med parent model info, applied recommendations og outcomes

    if len(recommendations) == 0:        """

        recommendations.append({        history = {

            'priority': 'low',            'has_parent': False,

            'title': 'Fine-tune Hyperparameters',            'parent_model': None,

            'why': 'Din model performer godt, men der er potentiale for marginal forbedring',            'version': training_log.get('version', 1),

            'action': 'Kør grid search for at finde optimale hyperparameters',            'applied_recommendation': None,

            'expected': '5-10% forbedring i validation accuracy'            'outcome': None,

        })            'related_models': []

            }

    return recommendations        

        # Check if this model was created from a recommendation

        parent_model_id = training_log.get('parent_model')

def check_model_history(model_id: str, current_log: Dict[str, Any]) -> Dict[str, Any]:        if parent_model_id:

    """Check if this model has a parent (previous version) and compare performance"""            history['has_parent'] = True

                history['parent_model'] = parent_model_id

    symbol = current_log.get('symbol', 'N/A')            history['applied_recommendation'] = training_log.get('applied_recommendation', {})

    model_type = current_log.get('model_type', 'unknown')            

    current_version = current_log.get('version', 1)            # Try to load parent model to compare

                try:

    history = {                parent_log_file = f"logs/{parent_model_id}_training.json"

        'version': current_version,                if os.path.exists(parent_log_file):

        'has_parent': False,                    with open(parent_log_file, 'r') as f:

        'related_models': []                        parent_log = json.load(f)

    }                    

                        # Calculate outcome (did recommendation help or hurt?)

    # Find all related models                    parent_val_mae = parent_log.get('metrics', {}).get('val_mae', 0)

    logs_dir = "logs"                    current_val_mae = training_log.get('metrics', {}).get('val_mae', 0)

    if os.path.exists(logs_dir):                    

        for log_file in os.listdir(logs_dir):                    if parent_val_mae > 0 and current_val_mae > 0:

            if log_file.endswith('_training.json'):                        improvement = ((parent_val_mae - current_val_mae) / parent_val_mae) * 100

                with open(os.path.join(logs_dir, log_file), 'r') as f:                        

                    other_log = json.load(f)                        history['outcome'] = {

                                            'parent_val_mae': parent_val_mae,

                if (other_log.get('symbol') == symbol and                             'current_val_mae': current_val_mae,

                    other_log.get('model_type') == model_type):                            'improvement_percent': improvement,

                                                'success': improvement > 0,

                    metrics = other_log.get('final_metrics', other_log.get('metrics', {}))                            'verdict': self._get_outcome_verdict(improvement)

                                            }

                    history['related_models'].append({            except Exception as e:

                        'version': other_log.get('version', 1),                print(f"{Fore.YELLOW}⚠️ Kunne ikke loade parent model: {str(e)}")

                        'val_mae': metrics.get('val_mae', 0),        

                        'description': other_log.get('description', '')        # Find all related models (same symbol + model_type)

                    })        model_type = training_log.get('model_type')

            symbol = training_log.get('symbol')

    # Sort by version        

    history['related_models'] = sorted(history['related_models'], key=lambda x: x['version'])        if model_type and symbol:

                try:

    # Check if there's a parent (previous version)                logs_dir = "logs"

    if current_version > 1:                related = []

        history['has_parent'] = True                

                        for log_file in os.listdir(logs_dir):

        # Find parent                    if not log_file.endswith('_training.json'):

        parent = None                        continue

        for related in history['related_models']:                    

            if related['version'] == current_version - 1:                    log_path = os.path.join(logs_dir, log_file)

                parent = related                    try:

                break                        with open(log_path, 'r') as f:

                                    log_data = json.load(f)

        if parent:                        

            current_metrics = current_log.get('final_metrics', current_log.get('metrics', {}))                        if (log_data.get('model_type') == model_type and 

            current_val_mae = current_metrics.get('val_mae', 0)                            log_data.get('symbol') == symbol):

            parent_val_mae = parent['val_mae']                            

                                        related.append({

            if parent_val_mae > 0:                                'model_id': log_file.replace('_training.json', ''),

                improvement = ((parent_val_mae - current_val_mae) / parent_val_mae) * 100                                'version': log_data.get('version', 1),

            else:                                'val_mae': log_data.get('metrics', {}).get('val_mae', 0),

                improvement = 0                                'description': log_data.get('description', ''),

                                            'applied_recommendation': log_data.get('applied_recommendation')

            history['outcome'] = {                            })

                'parent_val_mae': parent_val_mae,                    except:

                'current_val_mae': current_val_mae,                        continue

                'improvement_percent': improvement,                

                'success': improvement > 0                # Sort by version

            }                related.sort(key=lambda x: x['version'])

                            history['related_models'] = related

            # Determine verdict                

            if improvement > 10:            except Exception as e:

                history['outcome']['verdict'] = 'major_improvement'                print(f"{Fore.YELLOW}⚠️ Kunne ikke loade related models: {str(e)}")

            elif improvement > 5:        

                history['outcome']['verdict'] = 'improved'        return history

            elif improvement > 0:    

                history['outcome']['verdict'] = 'minor_improvement'    def _get_outcome_verdict(self, improvement: float) -> str:

            elif improvement > -5:        """Giver en verdict baseret på improvement percentage"""

                history['outcome']['verdict'] = 'minimal_degradation'        if improvement > 10:

            elif improvement > -20:            return "major_improvement"

                history['outcome']['verdict'] = 'degraded'        elif improvement > 5:

            else:            return "improved"

                history['outcome']['verdict'] = 'severely_degraded'        elif improvement > 0:

                        return "minor_improvement"

            # Check if there was an applied recommendation        elif improvement > -5:

            if 'applied_recommendation' in current_log:            return "minimal_degradation"

                history['applied_recommendation'] = current_log['applied_recommendation']        elif improvement > -20:

                return "degraded"

    return history        else:

            return "severely_degraded"

    

class MLMentorEngine:    def _get_llm_recommendations(self, analysis: Dict, training_log: Dict, model_history: Dict = None) -> List[Dict]:

    """        """

    ML Mentor Engine class for model analysis        Få anbefalinger fra LLM (OpenAI GPT-4) med historik kontekst

    (Legacy compatibility - use analyze_saved_model function instead)        """

    """        print(f"{Fore.CYAN}🤖 Spørger LLM om anbefalinger...")

            

    def __init__(self, api_key: Optional[str] = None):        try:

        self.api_key = api_key            # Build prompt with history

                prompt = self._build_mentor_prompt(analysis, training_log, model_history)

    def analyze_model(self, model_id: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:            

        """Analyze a model"""            # Call OpenAI API

        return analyze_saved_model(model_id, self.api_key, model)            response = self.client.chat.completions.create(

                model=self.model,  # Use configured model (gpt-4o-mini, gpt-4, gpt-4o)
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert ML mentor specializing in time series forecasting. Learn from past attempts - if a recommendation was tried and made the model worse, suggest different approaches. Provide concise, actionable advice with specific parameter recommendations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # Parse response
            llm_text = response.choices[0].message.content
            recommendations = self._parse_llm_response(llm_text, analysis)
            
            print(f"{Fore.GREEN}✅ LLM anbefalinger modtaget")
            return recommendations
            
        except Exception as e:
            print(f"{Fore.YELLOW}⚠️ LLM fejl: {str(e)}")
            print(f"{Fore.YELLOW}📝 Bruger rule-based fallback")
            return self._get_rule_based_recommendations(analysis)
    
    def _build_mentor_prompt(self, analysis: Dict, training_log: Dict, model_history: Dict = None) -> str:
        """
        Bygger prompt til LLM med historik kontekst
        """
        model_type = analysis['model_type']
        metrics = analysis['metrics']
        issues = analysis['issues']
        data_info = analysis['data_info']
        params = training_log.get('parameters', {})
        
        prompt = f"""Analyze this {model_type.upper()} stock prediction model:

📊 PERFORMANCE METRICS:
- Training MAE: ${metrics['train_mae']:.2f}
- Validation MAE: ${metrics['val_mae']:.2f}
- Generalization Gap: {metrics['generalization_gap']:.1f}%
- Training Samples: {data_info['training_samples']}

🔧 CURRENT PARAMETERS:
{json.dumps(params, indent=2)}

⚠️ DETECTED ISSUES:
"""
        
        for issue in issues:
            prompt += f"- [{issue['severity'].upper()}] {issue['description']}: {issue['impact']}\n"
        
        if not issues:
            prompt += "- None detected\n"
        
        # Add history context if available
        if model_history and model_history.get('has_parent'):
            prompt += "\n📜 PREVIOUS ATTEMPT HISTORY:\n"
            
            outcome = model_history.get('outcome')
            if outcome:
                applied_rec = model_history.get('applied_recommendation', {})
                rec_title = applied_rec.get('title', 'Unknown')
                rec_action = applied_rec.get('action', 'Unknown action')
                
                improvement = outcome['improvement_percent']
                verdict = outcome['verdict']
                
                prompt += f"- This is version {model_history['version']} (parent: v{model_history['version']-1})\n"
                prompt += f"- Previous recommendation applied: '{rec_title}'\n"
                prompt += f"- Action taken: {rec_action}\n"
                prompt += f"- OUTCOME: {improvement:+.1f}% change in Val MAE\n"
                
                if verdict == "major_improvement":
                    prompt += "  ✅ MAJOR SUCCESS - This approach worked very well!\n"
                elif verdict == "improved":
                    prompt += "  ✅ SUCCESS - This approach improved the model\n"
                elif verdict == "minor_improvement":
                    prompt += "  ✅ MINOR SUCCESS - Small improvement\n"
                elif verdict == "minimal_degradation":
                    prompt += "  ⚠️ MINIMAL HARM - Approach didn't help much\n"
                elif verdict == "degraded":
                    prompt += "  ❌ FAILED - This approach made the model WORSE\n"
                elif verdict == "severely_degraded":
                    prompt += "  🚨 MAJOR FAILURE - This approach severely damaged performance\n"
                
                prompt += f"- Old Val MAE: ${outcome['parent_val_mae']:.2f}\n"
                prompt += f"- New Val MAE: ${outcome['current_val_mae']:.2f}\n"
                
                # Add learning instruction
                if outcome['success']:
                    prompt += "\n💡 LEARNING: The previous approach worked! Consider similar optimizations.\n"
                else:
                    prompt += "\n💡 LEARNING: The previous approach FAILED! Avoid similar recommendations and try a DIFFERENT direction.\n"
            
            # Show related models if available
            related = model_history.get('related_models', [])
            if len(related) > 1:
                prompt += f"\n📊 Model Version History ({len(related)} versions):\n"
                for rel in related[-5:]:  # Show last 5 versions
                    desc = rel.get('description', 'Initial version')
                    val_mae = rel.get('val_mae', 0)
                    prompt += f"  - v{rel['version']}: {desc[:50]} (Val MAE: ${val_mae:.2f})\n"
        
        prompt += """

Based on ALL the above information (especially past failures), provide 2-3 specific, actionable recommendations:
1. [Priority: HIGH/MEDIUM/LOW] Recommendation title
   - Why: Brief explanation (mention if avoiding previous failed approach)
   - Action: Specific parameter change (e.g., "Reduce epochs from 50 to 35")
   - Expected: Predicted improvement

Be concise, specific, and encouraging. Focus on the most impactful changes first."""
        
        return prompt
    
    def _parse_llm_response(self, llm_text: str, analysis: Dict) -> List[Dict]:
        """
        Parser LLM response til strukturerede anbefalinger
        Håndterer både "### 1." og "1." format, samt "**Why:**" og "Why:" format
        """
        print(f"\n{Fore.CYAN}📝 LLM Response:")
        print(f"{Fore.WHITE}{llm_text}")
        print(f"{Fore.CYAN}{'='*60}\n")
        
        recommendations = []
        
        # Simple parsing - split by numbered lines
        lines = llm_text.split('\n')
        current_rec = None
        
        for line in lines:
            line = line.strip()
            
            # Remove markdown headers (### or **)
            line_clean = line.replace('###', '').replace('**', '').strip()
            
            # Check if line starts a new recommendation (numbered item with Priority marker)
            # Format: "1. [Priority: HIGH] Title" or "### 1. [Priority: HIGH] Title"
            is_new_recommendation = (
                line_clean and 
                line_clean[0].isdigit() and 
                '.' in line_clean and
                'Priority' in line_clean
            )
            
            if is_new_recommendation:
                # Save previous recommendation
                if current_rec and current_rec.get('title'):
                    recommendations.append(current_rec)
                
                # Extract priority
                priority = 'medium'
                if 'HIGH' in line_clean.upper():
                    priority = 'high'
                elif 'LOW' in line_clean.upper():
                    priority = 'low'
                
                # Extract title - everything after ']'
                title = line_clean
                if ']' in title:
                    title = title.split(']', 1)[-1].strip()
                else:
                    # Fallback: after number and dot
                    title = line_clean.split('.', 1)[-1].strip()
                
                current_rec = {
                    'priority': priority,
                    'title': title,
                    'why': '',
                    'action': '',
                    'expected': ''
                }
            
            # Extract details from bullet points
            # Format: "- Why: explanation" or "- **Why:** explanation"
            elif current_rec and line:
                # Clean markdown formatting
                line_clean_detail = line.replace('**', '').strip()
                
                # Check for detail markers (case insensitive)
                if line_clean_detail.startswith('-'):
                    # Remove leading dash and spaces
                    content = line_clean_detail[1:].strip()
                    content_lower = content.lower()
                    
                    if content_lower.startswith('why:'):
                        current_rec['why'] = content.split(':', 1)[1].strip()
                    elif content_lower.startswith('action:'):
                        current_rec['action'] = content.split(':', 1)[1].strip()
                    elif content_lower.startswith('expected:'):
                        current_rec['expected'] = content.split(':', 1)[1].strip()
                # Also check for indented bullet points
                elif current_rec and ('why:' in line.lower() or 'action:' in line.lower() or 'expected:' in line.lower()):
                    content = line.replace('**', '').replace('-', '').strip()
                    content_lower = content.lower()
                    
                    if content_lower.startswith('why:'):
                        current_rec['why'] = content.split(':', 1)[1].strip()
                    elif content_lower.startswith('action:'):
                        current_rec['action'] = content.split(':', 1)[1].strip()
                    elif content_lower.startswith('expected:'):
                        current_rec['expected'] = content.split(':', 1)[1].strip()
        
        # Add last recommendation
        if current_rec and current_rec.get('title'):
            recommendations.append(current_rec)
        
        # Fallback if parsing failed
        if not recommendations:
            recommendations = [{
                'priority': 'medium',
                'title': 'General Improvement',
                'why': 'Based on analysis',
                'action': llm_text[:200],  # First 200 chars
                'expected': 'Performance improvement'
            }]
        
        return recommendations
    
    def _get_rule_based_recommendations(self, analysis: Dict) -> List[Dict]:
        """
        Fallback til rule-based anbefalinger hvis LLM ikke er tilgængelig
        """
        print(f"{Fore.YELLOW}📝 Genererer rule-based anbefalinger...")
        
        recommendations = []
        issues = analysis['issues']
        metrics = analysis['metrics']
        
        # Recommendation 1: Fix overfitting
        overfitting_issues = [i for i in issues if i['type'] == 'overfitting']
        if overfitting_issues:
            severity = overfitting_issues[0]['severity']
            if severity == 'high':
                recommendations.append({
                    'priority': 'high',
                    'title': 'Reducer overfitting',
                    'why': f"Validation error er {metrics['generalization_gap']:.0f}% højere end training error",
                    'action': "Prøv: 1) Reducer epochs med 30%, 2) Tilføj regularization, 3) Øg training data",
                    'expected': "Forbedret generalization, lavere validation error"
                })
            else:
                recommendations.append({
                    'priority': 'medium',
                    'title': 'Optimer generalization',
                    'why': "Moderat gap mellem train og validation performance",
                    'action': "Reducer model kompleksitet eller tilføj mere training data",
                    'expected': "Bedre performance på nye data"
                })
        
        # Recommendation 2: Improve accuracy
        if metrics['val_mae'] > 10:
            recommendations.append({
                'priority': 'high' if metrics['val_mae'] > 15 else 'medium',
                'title': 'Forbedre prediction accuracy',
                'why': f"Validation MAE er ${metrics['val_mae']:.2f} - relativt højt",
                'action': "Eksperimenter med: 1) Længere window size, 2) Flere features, 3) Ensemble metoder",
                'expected': f"Reducer MAE til under ${max(5, metrics['val_mae'] * 0.7):.2f}"
            })
        
        # Recommendation 3: Data issues
        data_issues = [i for i in issues if i['type'] == 'insufficient_data']
        if data_issues:
            recommendations.append({
                'priority': 'high' if data_issues[0]['severity'] == 'high' else 'medium',
                'title': 'Øg training data',
                'why': f"Kun {analysis['data_info']['training_samples']} training samples",
                'action': "Download længere historisk periode (2-3 år anbefales)",
                'expected': "Mere robuste predictions, bedre generalization"
            })
        
        # If no issues, give optimization tips
        if not recommendations:
            recommendations.append({
                'priority': 'low',
                'title': 'Fin-tune hyperparameters',
                'why': "Modellen performer godt, men kan optimeres yderligere",
                'action': "Eksperimenter med learning rate, batch size, eller model arkitektur",
                'expected': "Små forbedringer i accuracy"
            })
        
        return recommendations[:3]  # Max 3 recommendations
    
    def _calculate_health_score(self, analysis: Dict) -> int:
        """
        Beregner overall health score (0-100)
        """
        score = 100
        
        # Deduct points for issues
        for issue in analysis['issues']:
            if issue['severity'] == 'high':
                score -= 20
            elif issue['severity'] == 'medium':
                score -= 10
            else:
                score -= 5
        
        # Bonus for strengths
        score += min(20, len(analysis['strengths']) * 5)
        
        return max(0, min(100, score))
    
    def auto_retrain_with_suggestions(self, model_id: str, training_log: Dict, 
                                     data, symbol: str) -> Dict:
        """
        Automatisk retrain med anbefalede parametre
        
        Returns:
            Dict med before/after comparison
        """
        print(f"\n{Fore.CYAN}🔄 Auto-retrain med anbefalinger...")
        
        # Get recommendations
        mentor_result = self.analyze_model(model_id, training_log)
        recommendations = mentor_result['recommendations']
        
        if not recommendations:
            print(f"{Fore.YELLOW}⚠️ Ingen anbefalinger at anvende")
            return None
        
        print(f"{Fore.GREEN}💡 Anvender {len(recommendations)} anbefalinger...")
        
        # Extract suggested parameters from recommendations
        suggested_params = self._extract_suggested_params(
            recommendations, 
            training_log.get('parameters', {})
        )
        
        print(f"{Fore.CYAN}📋 Foreslåede parametre:")
        for key, value in suggested_params.items():
            old_value = training_log.get('parameters', {}).get(key, 'N/A')
            print(f"   {key}: {old_value} → {value}")
        
        # TODO: Implement actual retraining with new params
        # This would call the appropriate train_and_save_* function
        
        return {
            'before': {
                'health_score': mentor_result['health_score'],
                'val_mae': mentor_result['analysis']['metrics']['val_mae']
            },
            'suggested_params': suggested_params,
            'recommendations': recommendations
        }
    
    def _extract_suggested_params(self, recommendations: List[Dict], 
                                  current_params: Dict) -> Dict:
        """
        Ekstraherer foreslåede parametre fra recommendations
        """
        import re
        suggested = current_params.copy()
        
        for rec in recommendations:
            action = rec.get('action', '').lower()
            title = rec.get('title', '').lower()
            
            # Extract n_estimators (RF/XGBoost)
            if 'n_estimators' in action or 'n_estimators' in title:
                numbers = re.findall(r'n_estimators[^\d]*(\d+)', action)
                if not numbers:
                    numbers = re.findall(r'to\s+(\d+)', action)
                if numbers:
                    suggested['n_estimators'] = int(numbers[0])
            
            # Extract max_depth
            if 'max_depth' in action or 'max_depth' in title:
                numbers = re.findall(r'max_depth[^\d]*(\d+)', action)
                if not numbers:
                    numbers = re.findall(r'to\s+(\d+)', action)
                if numbers:
                    suggested['max_depth'] = int(numbers[0])
            
            # Extract epochs (LSTM)
            if 'epoch' in action or 'epoch' in title:
                numbers = re.findall(r'epoch[s]?[^\d]*(\d+)', action)
                if not numbers:
                    numbers = re.findall(r'to\s+(\d+)', action)
                if numbers:
                    suggested['epochs'] = int(numbers[0])
            
            # Extract learning rate
            if 'learning' in action and 'rate' in action:
                numbers = re.findall(r'(\d+\.?\d*)', action)
                if numbers:
                    suggested['learning_rate'] = float(numbers[0])
            
            # Extract data period/samples
            if 'sample' in action or 'data' in action or 'historical' in action:
                if '500' in action or 'least 500' in action:
                    suggested['period'] = '3y'  # 3 years should give ~750 samples
                elif '2-3' in action or '3 year' in action:
                    suggested['period'] = '3y'
                elif '2 year' in action:
                    suggested['period'] = '2y'
        
        return suggested


# Helper function to load and analyze a model
def analyze_saved_model(model_id: str, api_key: Optional[str] = None, model: str = "gpt-4o-mini") -> Dict:
    """
    Hjælpe-funktion til at analysere en gemt model
    
    Args:
        model_id: Model ID (e.g., 'lstm_AAPL_20251012_213103')
        api_key: OpenAI API key (optional)
        model: OpenAI model to use ('gpt-4o-mini', 'gpt-4', 'gpt-4o')
    
    Returns:
        Analysis results dict
    """
    # Load training log
    logs_dir = "logs"
    log_file = os.path.join(logs_dir, f"{model_id}_training.json")
    
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Training log ikke fundet: {log_file}")
    
    with open(log_file, 'r') as f:
        training_log = json.load(f)
    
    # Create mentor and analyze
    mentor = MLMentorEngine(api_key=api_key, model=model)
    result = mentor.analyze_model(model_id, training_log)
    
    return result


if __name__ == "__main__":
    # Test ML Mentor Engine
    print("=" * 60)
    print("🧪 TESTING ML MENTOR ENGINE")
    print("=" * 60)
    
    # Find a training log to test with
    logs_dir = "logs"
    if os.path.exists(logs_dir):
        log_files = [f for f in os.listdir(logs_dir) if f.endswith('_training.json')]
        
        if log_files:
            # Use first log file
            log_file = log_files[0]
            model_id = log_file.replace('_training.json', '')
            
            print(f"\n📂 Testing med: {model_id}")
            
            try:
                result = analyze_saved_model(model_id)
                
                print(f"\n📊 RESULTS:")
                print(f"Health Score: {result['health_score']}/100")
                print(f"\nIssues Found: {len(result['analysis']['issues'])}")
                for issue in result['analysis']['issues']:
                    print(f"  - [{issue['severity'].upper()}] {issue['description']}")
                
                print(f"\nRecommendations: {len(result['recommendations'])}")
                for i, rec in enumerate(result['recommendations'], 1):
                    print(f"\n{i}. [{rec['priority'].upper()}] {rec['title']}")
                    print(f"   Why: {rec['why']}")
                    print(f"   Action: {rec['action']}")
                
                print(f"\n{Fore.GREEN}✅ ML Mentor Engine test gennemført!")
                
            except Exception as e:
                print(f"{Fore.RED}❌ Fejl: {str(e)}")
        else:
            print(f"{Fore.YELLOW}⚠️ Ingen training logs fundet i {logs_dir}")
    else:
        print(f"{Fore.RED}❌ Logs directory ikke fundet")
