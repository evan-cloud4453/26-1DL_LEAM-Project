import { Routes, Route } from 'react-router-dom'
import HomePage from './components/HomePage'
import StudySession from './components/StudySession'
import ReportView from './components/ReportView'
import HistoryView from './components/HistoryView'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/session" element={<StudySession />} />
      <Route path="/report/:sessionId" element={<ReportView />} />
      <Route path="/history" element={<HistoryView />} />
    </Routes>
  )
}
