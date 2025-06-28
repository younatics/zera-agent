#!/usr/bin/env python3
"""
배치로 여러 프롬프트 튜닝 실험을 실행하는 스크립트

Usage:
    python run_batch_experiments.py --config experiments_config.json
"""

import json
import subprocess
import time
import argparse
import os
from datetime import datetime
import logging
import sys
from pathlib import Path


# 평가 시스템 임포트
sys.path.append(str(Path(__file__).parent))
from evaluation.base.main import main as evaluation_main

def setup_logging():
    """로깅 설정"""
    log_file = f"batch_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # 커스텀 포맷터 클래스
    class ColoredFormatter(logging.Formatter):
        """색깔과 이모지가 포함된 로그 포맷터"""
        
        # ANSI 색깔 코드
        COLORS = {
            'DEBUG': '\033[36m',      # 청록색
            'INFO': '\033[32m',       # 녹색  
            'WARNING': '\033[33m',    # 노란색
            'ERROR': '\033[31m',      # 빨간색
            'CRITICAL': '\033[35m',   # 자주색
            'RESET': '\033[0m'        # 리셋
        }
        
        # 이모지 매핑
        EMOJIS = {
            'DEBUG': '🔍',
            'INFO': '📝',
            'WARNING': '⚠️',
            'ERROR': '❌',
            'CRITICAL': '🚨'
        }
        
        def format(self, record):
            # 시간 포맷
            time_str = self.formatTime(record, '%H:%M:%S')
            
            # 레벨별 색깔과 이모지
            level_color = self.COLORS.get(record.levelname, '')
            reset_color = self.COLORS['RESET']
            emoji = self.EMOJIS.get(record.levelname, '📝')
            
            # 메시지에 특별한 키워드가 있으면 추가 이모지
            message = record.getMessage()
            if '실험 시작:' in message:
                emoji = '🚀'
            elif '실험 완료:' in message:
                emoji = '✅'
            elif '실험 실패:' in message:
                emoji = '❌'
            elif '평가 시작' in message:
                emoji = '🔍'
            elif '평가 완료' in message:
                emoji = '🎯'
            elif '새로운 베스트' in message:
                emoji = '🏆'
            elif '진행도:' in message:
                emoji = '⏳'
            elif '대기' in message:
                emoji = '⏸️'
            elif '배치 실험 완료' in message:
                emoji = '🎉'
            
            # 포맷된 로그 메시지
            formatted = f"{level_color}{emoji} [{time_str}] {message}{reset_color}"
            return formatted
    
    # 콘솔 핸들러 (색깔 있음)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter())
    
    # 파일 핸들러 (색깔 없음)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # 루트 로거 설정
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def create_default_config():
    """기본 실험 설정 생성 - GSM8K 샘플 수 변화 실험"""
    # 타임스탬프 생성 (실행 시점 기준)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    config = {
        "experiments": [
            {
                "name": "GSM8K_Sample_5",
                "dataset": "gsm8k",
                "total_samples": 5,
                "iteration_samples": 5,
                "iterations": 10,
                "model": "local1",
                "evaluator": "local1",
                "meta_model": "local1",
                "output_dir": f"./results/gsm8k_sample_5_{timestamp}",
                "enabled": True
            },
            {
                "name": "GSM8K_Sample_20",
                "dataset": "gsm8k",
                "total_samples": 20,
                "iteration_samples": 5,
                "iterations": 10,
                "model": "local1",
                "evaluator": "local1",
                "meta_model": "local1",
                "output_dir": f"./results/gsm8k_sample_20_{timestamp}",
                "enabled": True
            },
            {
                "name": "GSM8K_Sample_50",
                "dataset": "gsm8k",
                "total_samples": 50,
                "iteration_samples": 5,
                "iterations": 10,
                "model": "local1",
                "evaluator": "local1",
                "meta_model": "local1",
                "output_dir": f"./results/gsm8k_sample_50_{timestamp}",
                "enabled": True
            },
            {
                "name": "GSM8K_Sample_100",
                "dataset": "gsm8k",
                "total_samples": 100,
                "iteration_samples": 5,
                "iterations": 10,
                "model": "local1",
                "evaluator": "local1",
                "meta_model": "local1",
                "output_dir": f"./results/gsm8k_sample_100_{timestamp}",
                "enabled": True
            },
            {
                "name": "GSM8K_Sample_200",
                "dataset": "gsm8k",
                "total_samples": 200,
                "iteration_samples": 5,
                "iterations": 10,
                "model": "local1",
                "evaluator": "local1",
                "meta_model": "local1",
                "output_dir": f"./results/gsm8k_sample_200_{timestamp}",
                "enabled": True
            }
        ],
        "global_settings": {
            "use_meta_prompt": True,
            "evaluation_threshold": 0.95,  # 높은 값으로 설정하여 거의 항상 개선되도록
            "score_threshold": None,
            "seed": 42,
            "delay_between_experiments": 5,  # 실험 간 대기 시간 (초)
            "run_evaluation": True,  # 실험 완료 후 평가 실행 여부
            "evaluation_samples": 500  # 평가용 샘플 수
        }
    }
    return config

def run_evaluation_after_experiment(experiment_config, output_dir, global_settings, logger):
    """실험 완료 후 GSM8K 평가 실행"""
    try:
        # 평가 실행 여부 확인
        if not global_settings.get("run_evaluation", True):
            logger.info("평가 실행이 비활성화되어 있습니다.")
            return True
        
        # 최신 config_*.json 파일 찾기 (모델 정보를 위해)
        config_files = list(Path(output_dir).glob("config_*.json"))
        if not config_files:
            logger.warning(f"설정 파일을 찾을 수 없습니다: {output_dir}")
            return False
        
        # 가장 최신 config 파일 선택
        latest_config = max(config_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"설정 파일 사용: {latest_config}")
        
        # config 파일에서 모델 정보 로드
        with open(latest_config, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # 최신 best_prompt_*.json 파일 찾기
        best_prompt_files = list(Path(output_dir).glob("best_prompt_*.json"))
        if not best_prompt_files:
            logger.warning(f"최고 성능 프롬프트 파일을 찾을 수 없습니다: {output_dir}")
            return False
        
        # 가장 최신 파일 선택
        latest_best_prompt = max(best_prompt_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"최고 성능 프롬프트 파일 사용: {latest_best_prompt}")
        
        # 최고 성능 프롬프트 로드
        with open(latest_best_prompt, 'r', encoding='utf-8') as f:
            best_prompt_data = json.load(f)
        
        system_prompt = best_prompt_data.get('system_prompt', '')
        user_prompt = best_prompt_data.get('user_prompt', '')
        avg_score = best_prompt_data.get('avg_score', 0.0)
        
        # 프롬프트 유효성 검사
        if not system_prompt or not user_prompt:
            logger.error("프롬프트가 비어있습니다. 평가를 건너뜁니다.")
            return False
        
        print(f"🏆 최고 성능 프롬프트 사용 (튜닝 점수: {avg_score:.3f})")
        print(f"📝 시스템 프롬프트: {len(system_prompt)} 문자")
        print(f"📝 유저 프롬프트: {len(user_prompt)} 문자")
        print(f"🤖 사용 모델: {config_data.get('model', 'solar')} (버전: {config_data.get('model_version', 'N/A')})")
        print(f"📊 평가 샘플 수: {global_settings.get('evaluation_samples', 500)}개")
        
        # api_client의 모델 정보 import
        from agent.common.api_client import Model as ApiModel
        
        # argparse.Namespace 객체 생성
        class EvaluationArgs:
            def __init__(self):
                self.dataset = "gsm8k"
                # config 파일에서 모델 정보 사용
                self.model = config_data.get("model", "local1")
                
                # config에서 model_version을 가져오고, 없으면 api_client의 기본값 사용
                model_name = config_data.get("model", "local1")
                self.model_version = config_data.get("model_version") or ApiModel.get_model_info(model_name)["default_version"]
                
                self.base_system_prompt = None
                self.base_user_prompt = None
                self.zera_system_prompt = system_prompt
                self.zera_user_prompt = user_prompt
                self.num_samples = global_settings.get("evaluation_samples", 500)
                self.temperature = 0.7
                self.top_p = 0.9
                self.base_num_shots = 0
                self.zera_num_shots = 0
                self.bbh_category = None
        
        eval_args = EvaluationArgs()
        
        # 평가 실행 및 결과 캡처
        print(f"🔍 GSM8K 평가 실행 중... (샘플 수: {eval_args.num_samples}개)")
        
        # 평가 결과를 파일로 캡처
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        eval_output_file = os.path.join(output_dir, f"evaluation_output_{timestamp}.txt")
        
        # 원래 sys.argv와 stdout 백업
        original_argv = sys.argv.copy()
        original_stdout = sys.stdout
        
        try:
            # stdout을 파일로 리다이렉트
            with open(eval_output_file, 'w', encoding='utf-8') as f:
                sys.stdout = f
                # evaluation_main 함수 호출
                evaluation_main(eval_args)
            
            # 평가 결과 파일에서 정확도 추출
            accuracy = extract_accuracy_from_output(eval_output_file)
            
            print(f"🎯 GSM8K 평가 완료!")
            print(f"📊 정확도: {accuracy:.2%}")
            print(f"💾 결과 파일: {eval_output_file}")
            
            # 평가 요약 정보 저장
            eval_summary = {
                "experiment_name": experiment_config["name"],
                "tuning_avg_score": avg_score,
                "gsm8k_accuracy": accuracy,
                "evaluation_samples": eval_args.num_samples,
                "timestamp": timestamp,
                "best_prompt_file": str(latest_best_prompt),
                "output_file": eval_output_file
            }
            
            eval_summary_file = os.path.join(output_dir, f"evaluation_summary_{timestamp}.json")
            with open(eval_summary_file, 'w', encoding='utf-8') as f:
                json.dump(eval_summary, f, ensure_ascii=False, indent=2)
            
            print(f"📋 요약 저장: {eval_summary_file}")
            return True
            
        except Exception as e:
            logger.error(f"평가 실행 중 오류: {str(e)}")
            return False
        finally:
            # stdout 복원
            sys.stdout = original_stdout
            # sys.argv 복원
            sys.argv = original_argv
            
    except Exception as e:
        logger.error(f"평가 준비 중 오류: {str(e)}")
        return False

def extract_accuracy_from_output(output_file):
    """평가 결과 파일에서 정확도 추출"""
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # "정확도: XX.XX%" 패턴 찾기
        import re
        accuracy_match = re.search(r'정확도:\s*([\d.]+)%', content)
        if accuracy_match:
            return float(accuracy_match.group(1)) / 100.0
        
        # "제라 프롬프트 정확도: XX.XX%" 패턴 찾기
        zera_accuracy_match = re.search(r'제라 프롬프트.*?정확도:\s*([\d.]+)%', content)
        if zera_accuracy_match:
            return float(zera_accuracy_match.group(1)) / 100.0
            
        return 0.0
        
    except Exception as e:
        print(f"정확도 추출 중 오류: {str(e)}")
        return 0.0

def print_experiment_header(experiment_name, experiment_num, total_experiments, logger):
    """실험 시작 헤더 출력"""
    print("\n" + "="*80)
    print(f"🚀 실험 {experiment_num}/{total_experiments}: {experiment_name}")
    print("="*80)

def print_section_divider(title, logger):
    """섹션 구분선 출력"""
    print(f"\n{'─'*60}")
    print(f"🔸 {title}")
    print("─"*60)

def run_experiment(experiment_config, global_settings, logger):
    """단일 실험 실행"""
    experiment_name = experiment_config["name"]
    
    # 출력 디렉토리 생성
    output_dir = experiment_config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # 명령어 구성
    cmd = [
        "python3", "run_prompt_tuning.py",
        "--dataset", experiment_config["dataset"],
        "--total_samples", str(experiment_config["total_samples"]),
        "--iteration_samples", str(experiment_config["iteration_samples"]),
        "--iterations", str(experiment_config["iterations"]),
        "--model", experiment_config["model"],
        "--evaluator", experiment_config["evaluator"],
        "--meta_model", experiment_config["meta_model"],
        "--output_dir", output_dir,
        "--seed", str(global_settings["seed"])
    ]
    
    # 글로벌 설정 추가
    if global_settings["use_meta_prompt"]:
        cmd.append("--use_meta_prompt")
    
    cmd.extend(["--evaluation_threshold", str(global_settings["evaluation_threshold"])])
    
    if global_settings["score_threshold"] is not None:
        cmd.extend(["--score_threshold", str(global_settings["score_threshold"])])
    
    logger.info(f"실행 명령어: {' '.join(cmd)}")
    
    print_section_divider("프롬프트 튜닝 실행", logger)
    print(f"🔥 명령어: {' '.join(cmd)}")
    
    # 실험 실행
    start_time = time.time()
    try:
        print("🚀 프롬프트 튜닝 시작...")
        print("─" * 60)
        print("📝 실시간 로그:")
        print("─" * 60)
        
        # 실시간 출력을 위해 capture_output=False로 변경
        result = subprocess.run(cmd, text=True, encoding='utf-8')
        end_time = time.time()
        duration = end_time - start_time
        
        print("─" * 60)
        if result.returncode == 0:
            print(f"✅ 프롬프트 튜닝 완료! (소요 시간: {duration:.1f}초)")
            
            # 실험 성공 시 평가 실행
            print_section_divider("GSM8K 평가 실행", logger)
            eval_success = run_evaluation_after_experiment(experiment_config, output_dir, global_settings, logger)
            if eval_success:
                print("🎯 평가 완료!")
            else:
                print("⚠️ 평가 실패")
                
        else:
            print(f"❌ 프롬프트 튜닝 실패! (리턴 코드: {result.returncode})")
            
        return result.returncode == 0, duration
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"❌ 실험 실행 중 예외 발생: {str(e)}")
        logger.error(f"예외 상세: {str(e)}")
        return False, duration

def main():
    parser = argparse.ArgumentParser(description="배치 프롬프트 튜닝 실험 실행")
    parser.add_argument("--config", type=str, default="experiments_config.json",
                       help="실험 설정 JSON 파일")
    parser.add_argument("--create_config", action="store_true",
                       help="기본 설정 파일 생성")
    parser.add_argument("--dry_run", action="store_true",
                       help="실제 실행 없이 명령어만 출력")
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    # 기본 설정 파일 생성
    if args.create_config:
        config = create_default_config()
        with open(args.config, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        logger.info(f"기본 설정 파일 생성: {args.config}")
        return
    
    # 설정 파일 로드 (없으면 기본 설정 사용)
    if os.path.exists(args.config):
        logger.info(f"설정 파일 사용: {args.config}")
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        logger.info("설정 파일이 없으므로 기본 설정을 사용합니다.")
        config = create_default_config()
    
    experiments = config["experiments"]
    global_settings = config["global_settings"]
    
    # 활성화된 실험만 필터링
    enabled_experiments = [exp for exp in experiments if exp.get("enabled", True)]
    
    logger.info(f"총 {len(enabled_experiments)}개의 실험을 실행합니다.")
    
    if args.dry_run:
        logger.info("=== DRY RUN MODE ===")
        for i, experiment in enumerate(enabled_experiments):
            logger.info(f"실험 {i+1}: {experiment['name']}")
            logger.info(f"  데이터셋: {experiment['dataset']}")
            logger.info(f"  전체 샘플: {experiment['total_samples']}")
            logger.info(f"  이터레이션 샘플: {experiment['iteration_samples']}")
            logger.info(f"  이터레이션: {experiment['iterations']}")
            logger.info(f"  모델: {experiment['model']}")
            logger.info(f"  출력 디렉토리: {experiment['output_dir']}")
        return
    
    # 배치 실험 시작 헤더
    print("\n" + "🎯" + "="*78 + "🎯")
    print(f"🎯 배치 프롬프트 튜닝 실험 시작! ({len(enabled_experiments)}개 실험)")
    print("🎯" + "="*78 + "🎯")
    
    # 실험 실행
    total_start_time = time.time()
    successful_experiments = 0
    failed_experiments = 0
    
    for i, experiment in enumerate(enabled_experiments):
        # 실험 헤더 출력
        print_experiment_header(experiment['name'], i+1, len(enabled_experiments), logger)
        
        # 실험 설정 요약 출력
        print(f"📊 데이터셋: {experiment['dataset']}")
        print(f"📈 총 샘플: {experiment['total_samples']}")
        print(f"🔄 이터레이션: {experiment['iterations']}")
        print(f"🤖 모델: {experiment['model']}")
        print(f"📁 출력: {experiment['output_dir']}")
        
        progress_percent = ((i) / len(enabled_experiments)) * 100
        progress_bar = "█" * int(progress_percent // 5) + "░" * (20 - int(progress_percent // 5))
        print(f"\n전체 진행률: [{progress_bar}] {progress_percent:.1f}%")
        
        success, duration = run_experiment(experiment, global_settings, logger)
        
        if success:
            successful_experiments += 1
            print(f"\n✅ 실험 완료! (소요시간: {duration:.1f}초)")
        else:
            failed_experiments += 1
            print(f"\n❌ 실험 실패! (소요시간: {duration:.1f}초)")
        
        # 다음 실험 전 대기 (마지막 실험이 아닌 경우)
        if i < len(enabled_experiments) - 1:
            delay = global_settings.get("delay_between_experiments", 60)
            print(f"\n⏸️ 다음 실험까지 {delay}초 대기...")
            
            # 카운트다운 표시
            for remaining in range(delay, 0, -1):
                print(f"\r⏳ 대기 중... {remaining}초 남음", end="", flush=True)
                time.sleep(1)
            print("\r" + " " * 30 + "\r", end="")  # 라인 클리어
    
    total_duration = time.time() - total_start_time
    
    # 최종 요약 헤더
    print("\n" + "🎉" + "="*78 + "🎉")
    print("🎉 배치 실험 완료!")
    print("🎉" + "="*78 + "🎉")
    
    # 실행 시간 포맷팅
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)
    
    if hours > 0:
        time_str = f"{hours}시간 {minutes}분 {seconds}초"
    elif minutes > 0:
        time_str = f"{minutes}분 {seconds}초"
    else:
        time_str = f"{seconds}초"
    
    # 최종 요약
    print(f"⏱️  총 실행 시간: {time_str}")
    print(f"✅ 성공한 실험: {successful_experiments}")
    print(f"❌ 실패한 실험: {failed_experiments}")
    print(f"📊 전체 실험: {len(enabled_experiments)}")
    
    # 성공률 계산 및 표시
    if len(enabled_experiments) > 0:
        success_rate = (successful_experiments / len(enabled_experiments)) * 100
        success_bar = "█" * int(success_rate // 5) + "░" * (20 - int(success_rate // 5))
        print(f"📈 성공률: [{success_bar}] {success_rate:.1f}%")
    
    # 평가 결과 요약 생성
    if global_settings.get("run_evaluation", True):
        print_section_divider("평가 결과 요약", logger)
        generate_evaluation_summary(enabled_experiments, logger)

def generate_evaluation_summary(experiments, logger):
    """모든 실험의 평가 결과 요약 생성"""
    summary_data = []
    
    for experiment in experiments:
        output_dir = experiment["output_dir"]
        
        # 평가 요약 파일 찾기
        eval_summary_files = list(Path(output_dir).glob("evaluation_summary_*.json"))
        if eval_summary_files:
            # 가장 최신 파일 선택
            latest_summary = max(eval_summary_files, key=lambda p: p.stat().st_mtime)
            
            try:
                with open(latest_summary, 'r', encoding='utf-8') as f:
                    eval_data = json.load(f)
                summary_data.append(eval_data)
            except Exception as e:
                logger.warning(f"평가 요약 파일 읽기 실패: {latest_summary} - {str(e)}")
    
    if summary_data:
        # 결과 정렬 (GSM8K 정확도 기준)
        summary_data.sort(key=lambda x: x.get("gsm8k_accuracy", 0), reverse=True)
        
        print("\n🏆 실험별 성능 순위:")
        print("─" * 80)
        print(f"{'순위':<4} {'실험명':<20} {'튜닝점수':<12} {'GSM8K정확도':<15} {'샘플수':<8}")
        print("─" * 80)
        
        for i, data in enumerate(summary_data, 1):
            exp_name = data.get("experiment_name", "N/A")[:18]
            tuning_score = data.get("tuning_avg_score", 0)
            gsm8k_acc = data.get("gsm8k_accuracy", 0)
            samples = data.get("evaluation_samples", 0)
            
            # 순위별 메달 이모지
            if i == 1:
                rank_emoji = "🥇"
            elif i == 2:
                rank_emoji = "🥈"
            elif i == 3:
                rank_emoji = "🥉"
            else:
                rank_emoji = f"{i:2d}"
            
            print(f"{rank_emoji:<4} {exp_name:<20} {tuning_score:<12.3f} {gsm8k_acc:<15.2%} {samples:<8}")
        
        print("─" * 80)
        
        # 최고 성능 하이라이트
        best_data = summary_data[0]
        print(f"\n🎖️  최고 성능: {best_data.get('experiment_name', 'N/A')}")
        print(f"   📊 GSM8K 정확도: {best_data.get('gsm8k_accuracy', 0):.2%}")
        print(f"   🎯 튜닝 점수: {best_data.get('tuning_avg_score', 0):.3f}")
        
        # 전체 요약 CSV 생성
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # results 폴더 확인/생성
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        summary_csv_file = results_dir / f"batch_evaluation_summary_{timestamp}.csv"
        
        try:
            import pandas as pd
            df = pd.DataFrame(summary_data)
            df.to_csv(summary_csv_file, index=False, encoding='utf-8')
            print(f"\n💾 전체 평가 요약 CSV 저장: {summary_csv_file}")
        except ImportError:
            logger.warning("⚠️ pandas가 설치되지 않아 CSV 파일 생성을 건너뜁니다.")
        except Exception as e:
            logger.error(f"❌ CSV 파일 생성 중 오류: {str(e)}")
    else:
        print("⚠️ 평가 결과 요약 데이터를 찾을 수 없습니다.")

if __name__ == "__main__":
    main() 