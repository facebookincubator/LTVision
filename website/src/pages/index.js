import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';

import Heading from '@theme/Heading';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/category/getting-started">
            Get Started
          </Link>
        </div>
      </div>
    </header>
  );
}


function HomePageCentralContent() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <div className="padding-vert--xl">
      <div className="container">
        <div className="row">
          <div className={clsx('col col--6', styles.descriptionSection)}>
            <h2>A foundational Open-Source Library for Lifetime Value (LTV) prediction</h2>
              <p className={styles.descriptionSectionText}>
                Best-in-class businesses are taking action and learning how they can incorporate their first-party data into their strategy now.
                Businesses who have right customer data strategy and infrastructure in place can start their journey to leverage exciting opportunity
                from data science methods which help predict future (long term) value of customer, referred as pLTV (predicted lifetime value) of customer. <br></br>
                A pLTV-focused approach to marketing  offers a significant advantage to  businesses over relying on short-term return on ad spend (ROAS)
              and cost tactics in terms of building a profitable business.
              </p>
          </div>
        </div>
      </div>
    </div>
  );
}


export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="Description will go into a meta tag in <head />">
      <HomepageHeader />
      <main>
        <HomePageCentralContent />
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
