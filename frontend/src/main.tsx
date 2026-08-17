import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { initAnalytics } from './lib/analytics'
import './index.css'

// Before render, and outside the component tree: Strict Mode double-invokes
// effects in development, and an analytics loader that runs twice reports
// every session twice.
initAnalytics()

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
