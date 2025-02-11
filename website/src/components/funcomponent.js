import React from 'react';
const funcomponent = () => {
  return (
    <div className="p-6 pb-0">
      <h3 className="mb-1.5 flex items-center gap-3 font-jakarta group-hover:text-primary">
        <svg
          className="h-7 w-7"
          fill="currentColor"
          aria-hidden="true"
          width="1em"
          height="1em"
          viewBox="0 0 20 20"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path d="M5 4a3 3 0 0 0-3 3v6a3 3 0 0 0 3 3h5a3 3 0 0 0 3-3v-.32l3.04 2.1c.83.57 1.96-.03 1.96-1.03v-7.5c0-1-1.13-1.6-1.96-1.03L13 7.32V7a3 3 0 0 0-3-3H5Zm8 4.54 3.6-2.5c.17-.1.4.01.4.21v7.5c0 .2-.23.32-.4.2L13 11.46V8.54ZM3 7c0-1.1.9-2 2-2h5a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7Z" />
        </svg>
        <div>Live Video</div>
      </h3>
      <p className="mb-0 text-sm text-zinc-400">
        Enable live video communication within your application. Perfect for education, telemedicine, social networks and gaming
      </p>
    </div>
  );
};
export default funcomponent;