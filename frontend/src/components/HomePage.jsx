import { useNavigate } from 'react-router-dom'
import './HomePage.css'

export default function HomePage() {
  const navigate = useNavigate()

  return (
    <div className="home-container">
      <div className="home-content">
        <h1 className="home-title">LEAM</h1>
        <p className="home-subtitle">Learning Environment Adaptive Model</p>
        <p className="home-desc">
          AI 기반 학습 집중도 분석 시스템
        </p>

        <div className="home-features">
          <div className="feature-card">
            <div className="feature-icon">👁</div>
            <h3>시선 추적</h3>
            <p>MediaPipe 기반 실시간 시선 및 얼굴 분석</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">📱</div>
            <h3>휴대폰 탐지</h3>
            <p>YOLOv8 기반 학습 방해 요소 감지</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">🔊</div>
            <h3>소음 분석</h3>
            <p>PANNs 기반 학습 환경 소음 분류</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">📊</div>
            <h3>학습 Zone</h3>
            <p>개인화된 학습 영역 적응형 판단</p>
          </div>
        </div>

        <div className="home-actions">
          <button className="btn-primary btn-large" onClick={() => navigate('/session')}>
            집중 모드 시작
          </button>
          <button className="btn-secondary btn-large" onClick={() => navigate('/history')}>
            학습 기록
          </button>
        </div>

        <div className="home-notice">
          <h4>안내 사항</h4>
          <ul>
            <li>웹캠과 마이크 접근 권한이 필요합니다.</li>
            <li>최초 3분간은 학습환경 분석 단계로 집중 시간이 누적되지 않습니다.</li>
            <li>얼굴이 장시간 검출되지 않을 경우 모드가 종료될 수 있습니다.</li>
          </ul>
        </div>
      </div>
    </div>
  )
}
